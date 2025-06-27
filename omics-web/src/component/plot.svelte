<script>
    import { createEventDispatcher, onMount, onDestroy, tick } from "svelte";
    import Plotly from "plotly.js-dist-min";
    import * as d3 from "d3";

    let spatialDiv;
    let clickedInfo;
    export let spatialData;
    export let image;
    // export let imageUrl;
    export let clusterColorScale;
    export let hoveredBarcode;
    export let lassoHover;
    export let lassoSelected = false;

    let clusterEdit = false;
    let availableClusters = [];
    let selectedCluster = null;
    let prevSpatialData;
    let comment = "";
    // let image;
    const dispatch = createEventDispatcher();
    let resizeObserver;
    let selectedBarcodes = [];
    let prevSelectedBarcodes = [];
    let plotInstance = null;
    let previewUrl = "";
    let annotationVisible = false;
    let annotationText = "";
    let annotationPos = {};
    let annotationColor = "";
    let previewBox, previewImg, previewOverlay, previewCircle;
    let lastSX = 0,
        lastSY = 0,
        lastSW = 0,
        lastSH = 0;
    let plotInitialized = false;
    let imageReady = false;
    let blob;

    function updatePreviewCircle(p) {
        if (
            !previewImg ||
            !previewCircle ||
            !image ||
            !plotInstance ||
            !previewUrl
        )
            return;

        tick().then(() => {
            const displayWidth = previewImg.clientWidth;
            const displayHeight = previewImg.clientHeight;

            const imageObj = plotInstance.layout.images?.[0];
            if (!imageObj) return;

            // 原图大小
            const imgW = image.width;
            const imgH = image.height;

            const sizex = imageObj.sizex ?? 1;
            const sizey = imageObj.sizey ?? 1;
            const x0Image = imageObj.x;
            const y0Image = imageObj.y;

            // 当前点在原图的像素坐标
            const relX = (p.x - x0Image) * (imgW / sizex);
            const relY = (p.y - y0Image) * (imgH / sizey);
            const flippedY = imgH - relY;

            // === 关键部分 ===
            // 你需要记住上次框选区域的 sx, sy, sw, sh，才能做映射
            const canvas = new Image();
            canvas.src = previewUrl;

            canvas.onload = () => {
                // 拿到裁剪的尺寸
                const sx = lastSX;
                const sy = lastSY;
                const sw = lastSW;
                const sh = lastSH;

                // 将原图上的点位置转换到裁剪图上的相对位置
                const clippedX = (relX - sx) / sw;
                const clippedY = (relY - sy) / sh;

                // clamp 限制在 0-1 区间
                const clampedX = Math.max(0, Math.min(1, clippedX));
                const clampedY = Math.max(0, Math.min(1, clippedY));

                const cx = clampedX * displayWidth;
                const cy = clampedY * displayHeight;

                previewOverlay.setAttribute("width", displayWidth);
                previewOverlay.setAttribute("height", displayHeight);
                previewCircle.setAttribute("cx", cx);
                previewCircle.setAttribute("cy", cy);
            };
        });
    }

    $: if (spatialData && imageReady && !plotInitialized) {
        drawPlot();
        plotInitialized = true;
    }

    function tryDrawPlot() {
        if (spatialData && imageReady) {
            drawPlot();
            plotInitialized = true;
        }
    }

    // 图像加载后才可绘制图层背景
    async function loadImage(url) {
        return new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.src = url;
            img.onload = () => resolve(img);
        });
    }

    // let prevSpatialData;
    let prevImage;
    // let plotInitialized = false;

    $: if (spatialData !== prevSpatialData) {
        prevSpatialData = spatialData;
        plotInitialized = false;
        tryDrawPlot();
    }

    $: if (image !== prevImage) {
        prevImage = image;
        plotInitialized = false;
        tryDrawPlot();
    }

    $: if (hoveredBarcode.from === "umap") {
        drawPlot();
    }

    function toBase64(img) {
        const canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        return canvas.toDataURL("image/png");
    }

    async function drawPlot() {
        console.log("Image size:", image.width, image.height);
        console.log(
            "Sample spatial point:",
            spatialData[0].x[0],
            spatialData[0].y[0],
        );
        console.log("image ready?", image.width, image.height);
        console.log("🎯 drawPlot:", {
            imageWidth: image.width,
            imageHeight: image.height,
            nPoints: spatialData.reduce((acc, t) => acc + t.x.length, 0),
        });
        if (plotInstance && spatialDiv) {
            Plotly.purge(spatialDiv);
            plotInstance = null;
        }
        // image = await loadImage(imageUrl);
        if (!image) {
            console.warn("❌ image not loaded yet");
            return;
        }
        const base64 = toBase64(image);
        const layout = {
            autosize: true,
            title: "Spatial Clusters",
            xaxis: {
                visible: false,
                range: [0, image.width], // ✅ 强制更新 x 轴范围
            },
            yaxis: {
                visible: false,
                range: [image.height, 0], // ✅ 强制更新 y 轴范围（注意 y 是反向）
                scaleanchor: "x",
                scaleratio: 1,
            },
            dragmode: false,
            margin: { l: 0, r: 0, t: 0, b: 0 },
            legend: { x: 0, y: 0, bgcolor: "rgba(255,255,255,0.6)" },
            images: [
                {
                    source: base64,
                    xref: "x",
                    yref: "y",
                    x: 0,
                    y: 0,
                    sizex: image.width,
                    sizey: image.height,
                    sizing: "contain",
                    xanchor: "left",
                    yanchor: "top",
                    opacity: 0.6,
                    layer: "below",
                },
            ],
        };

        const traces = spatialData.map((trace) => {
            const barcodes = trace.customdata || trace.text || [];
            const hoveredIndex = barcodes.indexOf(
                hoveredBarcode?.barcode ?? "",
            );

            const isHovering =
                hoveredBarcode?.barcode &&
                hoveredBarcode?.barcode !== "" &&
                hoveredBarcode?.barcode !== -1;

            let selectedIndices = null;
            if (lassoSelected && selectedBarcodes.length > 0) {
                selectedIndices = barcodes
                    .map((bc, i) => (selectedBarcodes.includes(bc) ? i : -1))
                    .filter((i) => i !== -1);
            }

            return {
                ...trace,
                marker: {
                    ...trace.marker,
                    color: clusterColorScale(trace.name),
                    opacity: isHovering ? 0.2 : 1, // 非高亮点透明
                },
                name: `Cluster ${trace.name}`,
                selectedpoints:
                    isHovering && hoveredIndex !== -1
                        ? [hoveredIndex]
                        : selectedIndices,
                selected: { marker: { opacity: 1 } },
                unselected: { marker: { opacity: 0.2 } },
            };
        });

        const clusterSet = new Set();
        spatialData.forEach((trace) => clusterSet.add(trace.name));
        availableClusters = Array.from(clusterSet);

        // ⚠️ 渲染并存下实例
        plotInstance = await Plotly.newPlot(spatialDiv, traces, layout, {
            displayModeBar: true,
            scrollZoom: true,
            displaylogo: false,
            modeBarButtons: [
                ["pan2d", "resetScale2d", "lasso2d", "select2d", "toImage"],
            ],
            responsive: true,
        });

        bindPlotEvents();
    }

    async function bindPlotEvents() {
        if (!plotInstance) return;

        plotInstance.on("plotly_selected", (eventData) => {
            (async () => {
                const lassoPaths = document.querySelectorAll(
                    ".selectionlayer path",
                );
                const lassoCircles = document.querySelectorAll(
                    ".outline-controllers circle",
                );
                lassoPaths.forEach((path) => path.remove());
                lassoCircles.forEach((circle) => circle.remove());
                clickedInfo = null;
                clusterEdit = false;
                lassoSelected = true;

                if (eventData?.points) {
                    if (eventData.range) {
                        const {
                            x: [x0, x1],
                            y: [y0, y1],
                        } = eventData.range;

                        const imageObj = plotInstance.layout.images?.[0];
                        if (!imageObj) {
                            console.warn("No layout.images found!");
                            return;
                        }

                        // Step 1: 图像逻辑空间中左上角
                        const x0Image = imageObj.x;
                        const y0Image = imageObj.y;

                        // Step 2: 选择框逻辑边界（Plotly坐标系）
                        const x0Sel = Math.min(x0, x1);
                        const x1Sel = Math.max(x0, x1);
                        const y0Sel = Math.min(y0, y1);
                        const y1Sel = Math.max(y0, y1);

                        // Step 3: 逻辑坐标到像素坐标缩放比
                        const scaleX = image.width / (imageObj.sizex ?? 1);
                        const scaleY = image.height / (imageObj.sizey ?? 1);

                        // Step 4: 转换到图像像素坐标系（Canvas）
                        const sx = (x0Sel - x0Image) * scaleX;
                        const sw = (x1Sel - x0Image) * scaleX - sx;

                        const sy = (y0Sel - y0Image) * scaleY;
                        const sh = (y1Sel - y0Image) * scaleY - sy;

                        lastSX = sx;
                        lastSY = sy;
                        lastSW = sw;
                        lastSH = sh;

                        const sizex = imageObj.sizex ?? 1;
                        console.log("eventData.range:", eventData.range);
                        console.log("canvas draw params:", { sx, sy, sw, sh });
                        console.log(
                            "image dimensions",
                            image.width,
                            image.height,
                        );

                        // 创建 canvas 并画图
                        const canvas = document.createElement("canvas");
                        canvas.width = sw;
                        canvas.height = sh;
                        const ctx = canvas.getContext("2d");
                        ctx.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);

                        previewUrl = canvas.toDataURL();
                        blob = base64ToBlob(previewUrl);
                    }

                    const barcodes = eventData.points.map(
                        (pt) => pt.customdata,
                    );
                    selectedBarcodes = barcodes;
                    prevSelectedBarcodes = barcodes;
                    console.log("Selected barcodes:", barcodes);

                    plotInstance.data.forEach((_, i) => {
                        Plotly.restyle(
                            plotInstance,
                            {
                                "selected.marker.opacity": [1],
                                "unselected.marker.opacity": [0.2],
                            },
                            [i],
                        );
                    });

                    setTimeout(() => {
                        document
                            .querySelectorAll(".selectionlayer path")
                            .forEach((el) => {
                                el.setAttribute(
                                    "style",
                                    el
                                        .getAttribute("style")
                                        ?.replace(
                                            /pointer-events:\s*[^;]+;?/g,
                                            "",
                                        )
                                        ?.replace(/cursor:\s*[^;]+;?/g, "") ??
                                        "",
                                );

                                el.style.pointerEvents = "none";
                                el.style.cursor = "default";
                            });
                    }, 0);

                    dispatch("spotClick", {
                        info: barcodes,
                        lassoSelected: true,
                        previewUrl: blob,
                    });
                }
            })();
        });

        plotInstance.on("plotly_deselect", () => {
            clickedInfo = null;
            lassoSelected = false;
            dispatch("spotClick", {
                info: clickedInfo,
                lassoSelected: lassoSelected,
            });
            clusterEdit = false;
            const lassoPaths = document.querySelectorAll(
                ".selectionlayer path",
            );
            const lassoCircles = document.querySelectorAll(
                ".outline-controllers circle",
            );
            const lassoRects = document.querySelectorAll(
                ".outline-controllers rect",
            );
            lassoPaths.forEach((path) => path.remove());
            lassoCircles.forEach((circle) => circle.remove());
            lassoRects.forEach((rect) => rect.remove());
            selectedBarcodes = [];
        });

        plotInstance.on("plotly_click", async (eventData) => {
            const mode = plotInstance._fullLayout.dragmode;
            clusterEdit = false;

            const point = eventData.points[0];
            const barcode = point.customdata;
            clickedInfo = { barcode, loading: true };

            selectedCluster = point.data.name;
            clickedInfo = {
                barcode,
                x: point.x,
                y: point.y,
                cluster: point.data.name.replace(/^cluster\s*/i, ""),
            };

            Plotly.relayout(spatialDiv, {
                annotations: [
                    {
                        x: point.x,
                        y: point.y,
                        text: `${barcode}`,
                        showarrow: true,
                        arrowhead: 1,
                        ax: 0,
                        ay: -40,
                        bgcolor: "white",
                        bordercolor: "",
                        borderwidth: 1,
                        layer: "above",
                    },
                ],
            });

            dispatch("spotClick", {
                info: clickedInfo,
                lassoSelected: lassoSelected,
            });
        });

        function approxEqual(a, b, tol = 1e-2) {
            return Math.abs(a - b) < tol;
        }

        plotInstance.on("plotly_relayout", (eventData) => {
            console.log("plotly_relayout triggered:", eventData);
            // const xRange = eventData["xaxis.range"];
            // const yRange = eventData["yaxis.range"];

            // const x0 = eventData["xaxis.range[0]"];
            // const x1 = eventData["xaxis.range[1]"];
            // const y0 = eventData["yaxis.range[0]"];
            // const y1 = eventData["yaxis.range[1]"];

            function approxEqual(a, b, tol = 1e-2) {
                return Math.abs(a - b) < tol;
            }

            const xRange = eventData["xaxis.range"] || [
                eventData["xaxis.range[0]"],
                eventData["xaxis.range[1]"],
            ];
            const yRange = eventData["yaxis.range"] || [
                eventData["yaxis.range[0]"],
                eventData["yaxis.range[1]"],
            ];

            if (
                eventData["yaxis.range"] &&
                eventData["yaxis.range"][1] === 0 &&
                eventData["yaxis.range"][0] === image.height
            ) {
                console.log("1111");
                plotInstance.data.forEach((_, i) => {
                    Plotly.restyle(
                        plotInstance,
                        {
                            selectedpoints: [null],
                            "selected.marker.opacity": [1],
                            "unselected.marker.opacity": [1],
                        },
                        [i],
                    );
                });

                Plotly.relayout(plotInstance, {
                    dragmode: false,
                    annotations: [],
                });
                annotationVisible = false;
                annotationText = "";
                annotationPos = {};

                const lassoPaths = document.querySelectorAll(
                    ".selectionlayer path",
                );
                const lassoCircles = document.querySelectorAll(
                    ".outline-controllers circle",
                );
                const lassoRects = document.querySelectorAll(
                    ".outline-controllers rect",
                );

                lassoPaths.forEach((path) => path.remove());
                lassoCircles.forEach((circle) => circle.remove());
                lassoRects.forEach((rect) => rect.remove());

                previewUrl = "";

                clickedInfo = null;
                lassoSelected = false;
                dispatch("spotClick", {
                    info: null,
                    lassoSelected: false,
                });
                let hoverInfo = {
                    barcode: -1,
                    from: "spotPlot",
                };
                dispatch("hover", hoverInfo);
                hoverInfo = {
                    barcode: -1,
                    from: "umap",
                };
                dispatch("hover", hoverInfo);
            }
        });

        plotInstance.on("plotly_hover", (eventData) => {
            const point = eventData.points?.[0];
            if (point) {
                const traceName = point.data.name; // "Cluster 0"

                const legendTexts = spatialDiv.querySelectorAll(".legendtext");

                legendTexts.forEach((textEl) => {
                    const label = textEl?.textContent?.trim();
                    if (label === traceName) {
                        textEl.style.fontWeight = "bold";
                        textEl.style.fill = "black";
                        textEl.parentNode.style.opacity = "1";
                    } else {
                        textEl.style.fontWeight = "normal";
                        textEl.style.fill = "#aaa";
                        textEl.parentNode.style.opacity = "0.3";
                    }
                });

                dispatch("hover", {
                    barcode: point.customdata,
                    from: "spotPlot",
                });
            }
        });

        plotInstance.on("plotly_unhover", () => {
            const legendTexts = spatialDiv.querySelectorAll(".legendtext");
            legendTexts.forEach((textEl) => {
                textEl.style.fontWeight = "normal";
                textEl.style.fill = "#444";
                textEl.parentNode.style.opacity = "1";
            });

            dispatch("hover", {
                barcode: -1,
                from: "spotPlot",
            });
        });

        window.addEventListener("resize", () => {
            Plotly.Plots.resize(spatialDiv);
        });
    }

    $: if (hoveredBarcode?.from === "umap") {
        const hovered = hoveredBarcode.barcode;

        const legendGroups = spatialDiv?.querySelectorAll(".traces") || [];

        if (!hovered || hovered === -1) {
            legendGroups.forEach((group) => {
                const textEl = group.querySelector(".legendtext");
                const pointEl = group.querySelector(".legendpoints path");

                textEl.style.fontWeight = "normal";
                textEl.style.fill = "#444";
                group.style.opacity = "1";
                if (pointEl) pointEl.style.opacity = "1";
            });
        } else if (plotInstance) {
            const match = plotInstance.data.find((trace) =>
                (trace.customdata || trace.text || []).includes(hovered),
            );
            if (match) {
                const traceName = match.name;

                legendGroups.forEach((group) => {
                    const textEl = group.querySelector(".legendtext");
                    const pointEl = group.querySelector(".legendpoints path");

                    const label = textEl?.textContent?.trim();
                    const isMatch = label === traceName;

                    textEl.style.fontWeight = isMatch ? "bold" : "normal";
                    textEl.style.fill = isMatch ? "black" : "#aaa";
                    group.style.opacity = isMatch ? "1" : "0.3";
                    if (pointEl) pointEl.style.opacity = isMatch ? "1" : "0.3";
                });
            }
        }
    }

    function findPointByBarcode(barcode) {
        for (let i = 0; i < spatialData.length; i++) {
            const trace = spatialData[i];
            const barcodes = trace.customdata || trace.text || [];

            const index = barcodes.indexOf(barcode);
            if (index !== -1) {
                return {
                    traceIndex: i,
                    pointIndex: index,
                    x: trace.x[index],
                    y: trace.y[index],
                    cluster: trace.name,
                };
            }
        }
        return null;
    }

    $: if (lassoHover && spatialDiv) {
        const p = findPointByBarcode(lassoHover.barcode);
        console.log(p);
        if (p) {
            Plotly.relayout(spatialDiv, {
                annotations: [
                    {
                        x: p.x,
                        y: p.y,
                        text: `${p.cluster}->${lassoHover.newCluster}`,
                        showarrow: true,
                        arrowhead: 1,
                        ax: 0,
                        ay: -40,
                        bgcolor: clusterColorScale(lassoHover.newCluster),
                        bordercolor: "",
                        borderwidth: 1,
                        layer: "above",
                    },
                ],
            });
            updatePreviewCircle(p);
        } else {
            Plotly.relayout(spatialDiv, {
                annotations: [],
            });
        }
        // const p = spatialData.
    }

    $: if (image) {
        if (image.complete) {
            imageReady = true;
        } else {
            image.onload = () => {
                imageReady = true;
            };
        }
    }

    onMount(() => {
        resizeObserver = new ResizeObserver(() => {
            if (plotInstance && spatialDiv) {
                Plotly.Plots.resize(spatialDiv);
            }
        });
        if (spatialDiv) resizeObserver.observe(spatialDiv);
    });

    onDestroy(() => {
        if (resizeObserver && spatialDiv) {
            resizeObserver.unobserve(spatialDiv);
        }
    });

    function base64ToBlob(base64Data, contentType = "image/png") {
        const byteCharacters = atob(base64Data.split(",")[1]);
        const byteNumbers = new Array(byteCharacters.length)
            .fill()
            .map((_, i) => byteCharacters.charCodeAt(i));
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: contentType });
    }
</script>

<div class="relative w-full h-full">
    <div class="h-full" bind:this={spatialDiv}></div>

    {#if previewUrl}
        <div
            class="absolute top-2 left-2 z-10 bg-white p-1 border border-gray-300 max-w-[300px] max-h-[300px] overflow-hidden"
            bind:this={previewBox}
        >
            <img
                src={previewUrl}
                alt="Preview"
                class="max-w-full max-h-full object-contain block"
                bind:this={previewImg}
            />

            <svg
                class="absolute top-0 left-0 pointer-events-none"
                xmlns="http://www.w3.org/2000/svg"
                style="width: 100%; height: 100%;"
                bind:this={previewOverlay}
            >
                <circle
                    r="6"
                    fill="none"
                    stroke="red"
                    stroke-width="2"
                    bind:this={previewCircle}
                />
            </svg>
        </div>
    {/if}
</div>

<style>
</style>

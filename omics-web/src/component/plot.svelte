<script>
    import { createEventDispatcher, onMount, onDestroy, tick } from "svelte";
    import Plotly from "plotly.js-dist-min";
    import * as d3 from "d3";

    let spatialDiv;
    let clickedInfo;
    export let spatialData;
    export let imageUrl;
    export let clusterColorScale;
    export let hoveredBarcode;
    export let lassoHover;
    export let lassoSelected = false;

    let clusterEdit = false;
    let availableClusters = [];
    let selectedCluster = null;
    let prevSpatialData;
    let comment = "";
    let image;
    const dispatch = createEventDispatcher();
    let resizeObserver;
    let selectedBarcodes = [];
    let prevSelectedBarcodes = [];
    let plotInstance = null;

    // 图像加载后才可绘制图层背景
    async function loadImage(url) {
        return new Promise((resolve) => {
            const img = new Image();
            img.src = url;
            img.onload = () => resolve(img);
        });
    }

    $: if (spatialData !== prevSpatialData) {
        prevSpatialData = spatialData;
        if (spatialData && image) {
            drawPlot();
        }
    }

    $: if (spatialData && !plotInstance) {
        selectedBarcodes = lassoSelected ? prevSelectedBarcodes : [];
        drawPlot();
    }

    $: if (hoveredBarcode.from === "umap") {
        drawPlot();
    }

    async function drawPlot() {
        image = await loadImage(imageUrl);
        const layout = {
            title: "Spatial Clusters",
            xaxis: { visible: false },
            yaxis: {
                visible: false,
                autorange: "reversed",
                scaleanchor: "x",
                scaleratio: 1,
            },
            dragmode: false,
            margin: { l: 0, r: 0, t: 0, b: 0 },
            legend: { x: 0, y: 0, bgcolor: "rgba(255,255,255,0.6)" },
            images: [
                {
                    source: imageUrl,
                    xref: "x",
                    yref: "y",
                    x: 0,
                    y: 0,
                    sizex: image.width,
                    sizey: image.height,
                    sizing: "stretch",
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

                    dispatch("spotClick", {
                        info: barcodes,
                        lassoSelected: lassoSelected,
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
                    },
                ],
            });

            dispatch("spotClick", {
                info: clickedInfo,
                lassoSelected: lassoSelected,
            });
        });

        plotInstance.on("plotly_relayout", (eventData) => {
            if (
                eventData["xaxis.autorange"] === true &&
                eventData["yaxis.autorange"] === true
            ) {
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
                        bgcolor: "white",
                        bordercolor: "",
                        borderwidth: 1,
                    },
                ],
            });
        } else {
            Plotly.relayout(spatialDiv, {
                annotations: [],
            });
        }
        // const p = spatialData.
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
</script>

<div class="h-full" bind:this={spatialDiv}></div>

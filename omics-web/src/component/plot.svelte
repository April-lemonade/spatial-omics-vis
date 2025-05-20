<script>
    import { createEventDispatcher, onMount, tick } from "svelte";
    import Plotly from "plotly.js-dist-min";
    import * as d3 from "d3";

    let spatialDiv;
    let clickedInfo;
    export let spatialData;
    export let imageUrl;
    export let clusterColorScale;
    export let hoveredBarcode;
    let lassoSelected = false;

    let clusterEdit = false;
    let availableClusters = [];
    let selectedCluster = null;
    let comment = "";
    let image;
    const dispatch = createEventDispatcher();

    let plotInstance = null;

    // 图像加载后才可绘制图层背景
    async function loadImage(url) {
        return new Promise((resolve) => {
            const img = new Image();
            img.src = url;
            img.onload = () => resolve(img);
        });
    }

    // 监听 spatialData 一旦加载，开始绘图
    $: if (spatialData && image) {
        drawPlot();
    }

    $: if (hoveredBarcode.from === "umap") {
        drawPlot();
    }

    async function drawPlot() {
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

            return {
                ...trace,
                marker: {
                    ...trace.marker,
                    color: clusterColorScale(trace.name),
                    opacity: isHovering ? 0.2 : 1, // 非高亮点透明
                },
                name: `Cluster ${trace.name}`,
                selectedpoints:
                    isHovering && hoveredIndex !== -1 ? [hoveredIndex] : null,
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
            modeBarButtons: [["pan2d", "resetScale2d", "lasso2d", "select2d"]],
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
                cluster: point.data.name,
            };

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

                Plotly.relayout(plotInstance, { dragmode: false });

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
                const hoverInfo = {
                    barcode: -1,
                    from: "spotPlot",
                };
                dispatch("hover", hoverInfo);
            }
        });

        plotInstance.on("plotly_hover", (eventData) => {
            const point = eventData.points?.[0];
            if (point) {
                const hoverInfo = {
                    barcode: point.customdata,
                    from: "spotPlot",
                };
                dispatch("hover", hoverInfo);
            }
        });

        plotInstance.on("plotly_unhover", () => {
            const hoverInfo = {
                barcode: -1,
                from: "spotPlot",
            };
            dispatch("hover", hoverInfo);
        });
    }

    onMount(async () => {
        image = await loadImage(imageUrl);
    });
</script>

<div class="h-full" bind:this={spatialDiv}></div>

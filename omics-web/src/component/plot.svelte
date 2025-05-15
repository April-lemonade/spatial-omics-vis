<script>
    import { createEventDispatcher, onMount } from "svelte";
    import Plotly from "plotly.js-dist-min";
    import * as d3 from "d3";

    let spatialDiv;
    let clickedInfo;
    export let spatialData;
    export let imageUrl;
    export let clusterColorScale;
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
    $: if (spatialData && imageUrl) {
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

        const traces = spatialData.map((trace) => ({
            ...trace,
            marker: {
                ...trace.marker,
                color: clusterColorScale(trace.name), // 👈 明确指定颜色
            },
            selected: { marker: { opacity: 1 } },
            unselected: { marker: { opacity: 0.2 } },
        }));

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
                clickedInfo = null;
                clusterEdit = false;
                lassoSelected = true;

                // dispatch("spotClick", {
                //     info: clickedInfo,
                //     lassoSelected: lassoSelected,
                // });

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
                                "unselected.marker.opacity": [0.2], // 👈 你希望的淡化效果
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
            // plotInstance.data.forEach((_, i) => {
            //     Plotly.restyle(
            //         plotInstance,
            //         {
            //             "selected.marker.opacity": 1,
            //             "unselected.marker.opacity": 1,
            //         },
            //         [i],
            //     );
            // });
        });

        plotInstance.on("plotly_click", async (eventData) => {
            const mode = plotInstance._fullLayout.dragmode;
            // if(mode==)
            // if (mode === "lasso" || mode === "select") {
            //     // 禁用点击行为（正在套索模式中）
            //     return;
            // }
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
                            selectedpoints: [null], // null 是关键！不能是 [[]]
                            "selected.marker.opacity": [1],
                            "unselected.marker.opacity": [1],
                        },
                        [i],
                    );
                });

                const lassoPaths = document.querySelectorAll(
                    ".selectionlayer path",
                );
                const lassoCircles = document.querySelectorAll(
                    ".outline-controllers circle",
                );
                lassoPaths.forEach((path) => path.remove());
                lassoCircles.forEach((circle) => circle.remove());

                clickedInfo = null;
                lassoSelected = false;
                dispatch("spotClick", {
                    info: null,
                    lassoSelected: false,
                });
            }
        });

        window.addEventListener("resize", () => {
            Plotly.Plots.resize(plotInstance);
        });
    }
</script>

<div class="h-full" bind:this={spatialDiv}></div>

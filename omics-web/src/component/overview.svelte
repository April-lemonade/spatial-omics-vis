<script>
    import { createEventDispatcher, onMount, tick } from "svelte";
    import Plotly from "plotly.js-dist-min";

    import * as d3 from "d3";
    import { Tabs } from "@skeletonlabs/skeleton-svelte";
    import { all } from "three/tsl";

    export let spotMetricsData;
    export let clusterColorScale;
    export let allLog;
    export let currentSlice;
    export let baseApi;
    export let hoveredBarcode;
    export let hvg;

    const dispatch = createEventDispatcher();

    let violinDiv;
    let donutDiv;
    let umapDiv;
    let barChartDiv;
    let expandedIndex = null;
    let umap;
    let umapData;
    let enrichmentResults;

    let groups = ["Overview", "Cluster", "Gene", "Log"];
    let group = groups[0];

    const metrics = [
        "nCount_Spatial",
        "nFeature_Spatial",
        "percent_mito",
        "percent_ribo",
    ];

    // 每次切换 tab 到 Overview 且数据存在时画图
    $: if (
        group === "Overview" &&
        spotMetricsData &&
        violinDiv &&
        umapDiv &&
        umapData
    ) {
        tick().then(() => {
            drawFacetViolins(spotMetricsData);
            drawUMAP();
        });
    }

    $: if (hoveredBarcode.from === "spotPlot" && umapData) {
        // console.log(hoveredBarcode);
        drawUMAP();
    }

    $: if (group === "Cluster" && spotMetricsData && donutDiv) {
        tick().then(() => {
            // drawFacetViolins(spotMetricsData);
            drawDonut(spotMetricsData);
        });
    }

    function drawFacetViolins(data) {
        const traces = [];
        const allClusters = [...new Set(data.map((d) => d.cluster))].sort(
            (a, b) => +a - +b,
        );

        for (let i = 0; i < metrics.length; i++) {
            const metric = metrics[i];
            const axisSuffix = i === 0 ? "" : i + 1;

            const metricData = data.filter((d) => d.metric === metric);

            traces.push({
                type: "violin",
                x: Array(metricData.length).fill("All Spots"),
                y: metricData.map((d) => d.value),
                customdata: metricData.map((d) => d.barcode),
                hovertemplate: "Barcode: %{customdata}<extra></extra>",
                name: metric,
                box: { visible: true },
                meanline: { visible: true },
                points: false,
                box: { visible: true },
                meanline: { visible: true },
                jitter: 0.4,
                pointpos: 0,
                side: "both",
                marker: {
                    color: "black", // 点的颜色
                    size: 2.5, // 点的大小
                },
                fillcolor: "rgba(231, 76, 60, 0.6)", // violin填充色
                line: {
                    color: "rgba(231, 76, 60, 1)", // 边框色
                    width: 1,
                },
                xaxis: `x${axisSuffix}`,
                yaxis: `y${axisSuffix}`,
                showlegend: false,
            });
        }

        const layout = {
            grid: { rows: 2, columns: 2, pattern: "independent" },
            margin: { t: 40, l: 40, r: 20, b: 40 },
            title: "Spot-Level Metrics by Cluster (2×2 layout)",
            showlegend: false,
            annotations: metrics.map((metric, i) => {
                const col = i % 2;
                const row = Math.floor(i / 2);

                return {
                    text: metric,
                    font: { size: 16, color: "#333" },
                    showarrow: false,
                    xref: "paper",
                    yref: "paper",
                    x: col * 0.5 + 0.25, // 中间对齐当前列
                    y: 1 - row * 0.6, // 子图上方略微偏移
                    xanchor: "center",
                    yanchor: "bottom",
                };
            }),
        };

        // 设置每个子图的 x/y 轴
        for (let i = 0; i < metrics.length; i++) {
            const idx = i === 0 ? "" : i + 1;
            layout[`xaxis${idx}`] = {
                title: "",
                ticktext: [],
                tickvals: [],
                showticklabels: false,
            };
            layout[`yaxis${idx}`] = { title: metrics[i] };
        }

        Plotly.newPlot(violinDiv, traces, layout, {
            scrollZoom: true,
            responsive: true,
            useResizeHandler: true,
            displaylogo: false,
            modeBarButtons: [["pan2d", "resetScale2d", "toImage"]],
        });
    }

    async function drawDonut(data) {
        // 只用一次 cluster 数据，不用重复的 metric
        const clusterCounts = {};

        data.forEach((d) => {
            if (!clusterCounts[d.cluster]) clusterCounts[d.cluster] = 0;
            clusterCounts[d.cluster]++;
        });

        const clusters = Object.keys(clusterCounts).sort((a, b) => +a - +b);
        const values = clusters.map((c) => clusterCounts[c]);
        const labels = clusters.map((c) => ` Cluster ${+c}`); // +1 for display

        const colors = clusters.map((c) => clusterColorScale(c));
        const trace = {
            type: "pie",
            labels,
            values,
            hole: 0.5, // Donut hole
            marker: { colors },
            textinfo: "percent",
            hoverinfo: "label+value",
        };

        const layout = {
            title: "Spot Count by Cluster",
            margin: { l: 0, r: 0, t: 0, b: 0 },
            showlegend: false,
            autosize: true,
            width: donutDiv.clientWidth / 2,
        };

        let donut = await Plotly.newPlot(donutDiv, [trace], layout, {
            responsive: true,
            useResizeHandler: true,
            displaylogo: false,
            modeBarButtons: [[]],
        });

        window.addEventListener("resize", () => {
            Plotly.Plots.resize(donut);
        });
    }

    async function fecthUmapData() {
        const response = await fetch(
            baseApi + `/umap-coordinates?slice_id=${currentSlice}`,
        );

        const data = await response.json();
        return data;
    }

    async function drawUMAP() {
        const layout = {
            margin: { l: 40, r: 10, t: 30, b: 30 },
            height: 400,
            showlegend: false,
            autosize: true,
            width: umapDiv.clientWidth,
        };

        const grouped = Array.from(d3.group(umapData, (d) => d.cluster));

        const sortedGrouped = grouped.sort((a, b) => +a[0] - +b[0]);
        const traces = sortedGrouped.map(([cluster, points]) => {
            const barcodes = points.map((d) => d.barcode);
            const hoveredIndex = barcodes.indexOf(
                hoveredBarcode?.barcode ?? "",
            );

            // console.log(hoveredIndex);

            const isHovering =
                hoveredBarcode?.barcode &&
                hoveredBarcode?.barcode !== "" &&
                hoveredBarcode?.barcode !== -1;

            return {
                x: points.map((d) => d.UMAP_1),
                y: points.map((d) => d.UMAP_2),
                text: barcodes,
                name: `Cluster ${cluster}`,
                type: "scatter",
                mode: "markers",
                marker: {
                    color: clusterColorScale(cluster),
                    size: 4,
                    opacity: isHovering ? 0.1 : 1,
                },
                selectedpoints: !isHovering
                    ? null
                    : hoveredIndex !== -1
                      ? [hoveredIndex]
                      : [],
                selected: { marker: { opacity: 1 } },
                unselected: { marker: { opacity: 0.1 } },
                hovertemplate: "Barcode: %{text}<extra></extra>",
            };
        });

        // console.log(data);

        umap = await Plotly.newPlot(umapDiv, traces, layout, {
            scrollZoom: true,
            responsive: true,
            useResizeHandler: true,
            displaylogo: false,
            modeBarButtons: [["toImage"]],
        });

        window.addEventListener("resize", () => {
            Plotly.Plots.resize(umap);
        });

        umap.on("plotly_hover", (eventData) => {
            const point = eventData.points?.[0];
            if (point) {
                const hoverInfo = {
                    barcode: point.text,
                    from: "umap",
                };
                dispatch("hover", hoverInfo);
            }
        });

        umap.on("plotly_unhover", () => {
            const hoverInfo = {
                barcode: -1,
                from: "umap",
            };
            dispatch("hover", hoverInfo);
        });

        window.addEventListener("resize", () => {
            Plotly.Plots.resize(umap);
        });
    }

    $: if (hvg && barChartDiv && group === "Gene") {
        tick().then(() => {
            drawEnrichmentChart(hvg);
        });
    }

    function drawEnrichmentChart(results) {
        const clusters = [...new Set(results.map((d) => d.Category))];
        const colorScale = d3.scaleOrdinal(d3.schemeTableau10).domain(clusters);

        const maxLabelLength = 60;

        const yLabels = [...results].map((d) =>
            d.Term.length > maxLabelLength
                ? d.Term.slice(0, maxLabelLength - 3) + "..."
                : d.Term,
        );

        const tracesMap = new Map();

        // 构建每个 Category 的 trace
        for (const d of results) {
            const label =
                d.Term.length > maxLabelLength
                    ? d.Term.slice(0, maxLabelLength - 3) + "..."
                    : d.Term;

            if (!tracesMap.has(d.Category)) {
                tracesMap.set(d.Category, {
                    type: "bar",
                    name: d.Category,
                    x: [],
                    y: [],
                    orientation: "h",
                    text: [],
                    textposition: "none",
                    customdata: [],
                    hovertemplate:
                        "<b>%{text}</b><br>Category: " +
                        d.Category +
                        "<br>-log10(p-adj): %{x:.2f}<br>Genes: %{customdata}<extra></extra>",
                    marker: {
                        color: colorScale(d.Category),
                    },
                });
            }

            const trace = tracesMap.get(d.Category);
            trace.x.push(-Math.log10(d["Adjusted P-value"]));
            trace.y.push(label);
            trace.text.push(d.Term);
            trace.customdata.push(d.Genes.split(";").length);
        }

        const traces = [...tracesMap.values()];

        const layout = {
            showlegend: true,
            title: "Top Enriched Terms (Grouped by Category)",
            barmode: "group",
            margin: { l: 200, r: 0, t: 100, b: 20 },
            height: yLabels.length * 20,
            // autosize: true,
            // width: barChartDiv.clientWidth,
            yaxis: {
                categoryorder: "array",
                categoryarray: yLabels,
                automargin: true,
                tickfont: { size: 12 },
            },
            xaxis: {
                title: "-log10(Adjusted P-value)",
                tickfont: { size: 12 },
            },
            legend: {
                orientation: "h", // 横向排列
                x: -50,
                y: 10,
                xanchor: "center", // 居中对齐
            },
        };

        Plotly.newPlot(barChartDiv, traces, layout, {
            scrollZoom: false,
            responsive: true,
            useResizeHandler: true,
            displaylogo: false,
            modeBarButtons: [["pan2d", "resetScale2d", "toImage"]],
        });
    }

    onMount(async () => {
        umapData = await fecthUmapData();
    });
</script>

<Tabs
    bind:value={group}
    onValueChange={(e) => (group = e.value)}
    class="w-full h-full"
>
    {#snippet list()}
        {#each groups as g}
            <Tabs.Control value={g}>{g}</Tabs.Control>
        {/each}
    {/snippet}

    {#snippet content()}
        {#each groups as g}
            <Tabs.Panel value={g}>
                {#if g === "Overview"}
                    <div bind:this={umapDiv}></div>
                    <div bind:this={violinDiv}></div>
                {:else if g === "Cluster"}
                    <div bind:this={donutDiv} class="w-full max-w-full"></div>
                {:else if g === "Log" && allLog}
                    <div class="table-wrap">
                        <table class="table caption-bottom text-xs w-full">
                            <thead>
                                <tr>
                                    <th>Barcode</th>
                                    <th>Prev</th>
                                    <th>New</th>
                                </tr>
                            </thead>
                            <tbody class="[&>tr]:hover:preset-tonal-primary">
                                {#each allLog as row, i}
                                    <tr
                                        class="cursor-pointer"
                                        on:click={() =>
                                            (expandedIndex =
                                                expandedIndex === i ? null : i)}
                                    >
                                        <td>{row.barcode}</td>
                                        <td>{row.old_cluster}</td>
                                        <td>{row.new_cluster}</td>
                                    </tr>
                                    {#if expandedIndex === i}
                                        <tr class="bg-muted/30 text-sm">
                                            <td colspan="3">
                                                <div>
                                                    <strong>Comment:</strong>
                                                    {row.comment || "-"}
                                                </div>
                                                <div>
                                                    <strong>Time:</strong>
                                                    {row.updated_at}
                                                </div>
                                            </td>
                                        </tr>
                                    {/if}
                                {/each}
                            </tbody>
                            <tfoot>
                                <tr>
                                    <td colspan="2">Total</td>
                                    <td class="text-right">{allLog.length}</td>
                                </tr>
                            </tfoot>
                        </table>
                    </div>
                {:else}
                    <div bind:this={barChartDiv} class="w-full h-full"></div>
                {/if}
            </Tabs.Panel>
        {/each}
    {/snippet}
</Tabs>

<script>
    import Plotly from "plotly.js-dist-min";
    import { tick } from "svelte";
    import * as d3 from "d3";
    import { Tabs } from "@skeletonlabs/skeleton-svelte";
    import { all } from "three/tsl";

    export let spotMetricsData;
    export let clusterColorScale;
    export let allLog;
    let violinDiv;
    let donutDiv;

    let groups = ["Overview", "Cluster", "Log"];
    let group = groups[0];

    const metrics = [
        "nCount_Spatial",
        "nFeature_Spatial",
        "percent_mito",
        "percent_ribo",
    ];

    // 每次切换 tab 到 Overview 且数据存在时画图
    $: if (group === "Overview" && spotMetricsData && violinDiv) {
        tick().then(() => {
            drawFacetViolins(spotMetricsData);
            // drawDonut(spotMetricsData);
        });
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
            modeBarButtons: [["pan2d", "resetScale2d"]],
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
        const labels = clusters.map((c) => `Cluster ${+c}`); // +1 for display

        const colors = clusters.map((c) => clusterColorScale(`Cluster ${+c}`)); // 👈 统一格式
        const trace = {
            type: "pie",
            labels,
            values,
            hole: 0.5, // Donut hole
            marker: { colors },
            textinfo: "percent", // 不显示百分比
            hoverinfo: "label+value",
        };

        const layout = {
            title: "Spot Count by Cluster",
            margin: { l: 0, r: 0, t: 0, b: 0 },
            showlegend: true,
            autosize: true,
            width: donutDiv.clientWidth,
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
</script>

<Tabs bind:value={group} onValueChange={(e) => (group = e.value)}>
    {#snippet list()}
        {#each groups as g}
            <Tabs.Control value={g}>{g}</Tabs.Control>
        {/each}
    {/snippet}

    {#snippet content()}
        {#each groups as g}
            <Tabs.Panel value={g}>
                {#if g === "Overview"}
                    <div bind:this={violinDiv}></div>
                {:else if g === "Cluster"}
                    <div bind:this={donutDiv} class="w-full max-w-full"></div>
                {:else if allLog}
                    <div class="table-wrap ">
                        <table class="table caption-bottom text-xs">
                            <!-- <caption class="pt-4"
                                >A list of elements from the periodic table.</caption
                            > -->
                            <thead>
                                <tr>
                                    <th>Barcode</th>
                                    <th>Prev</th>
                                    <th>New</th>
                                    <th>Comment</th>
                                    <th>Time</th>
                                </tr>
                            </thead>
                            <tbody class="text-sm [&>tr]:hover:preset-tonal-primary">
                                {#each allLog as row}
                                    <tr>
                                        <td>{row.barcode}</td>
                                        <td>{row.old_cluster}</td>
                                        <td>{row.new_cluster}</td>
                                        <td>{row.comment}</td>
                                        <td>{row.updated_at}</td>
                                    </tr>
                                {/each}
                            </tbody>
                            <tfoot>
                                <tr>
                                    <td colspan="4">Total</td>
                                    <td class="text-right">{allLog.length}</td>
                                </tr>
                            </tfoot>
                        </table>
                    </div>
                    <!-- {#each allLog as l}
                        {#each Object.entries(l) as [key, value]}
                            <li>
                                {key}: {value}
                            </li>
                        {/each}
                    {/each} -->
                {/if}
            </Tabs.Panel>
        {/each}
    {/snippet}
</Tabs>

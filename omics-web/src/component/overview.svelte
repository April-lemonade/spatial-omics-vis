<script>
    import { createEventDispatcher, onMount, tick } from "svelte";
    import Plotly from "plotly.js-dist-min";

    import * as d3 from "d3";
    import { Tabs } from "@skeletonlabs/skeleton-svelte";
    import { all } from "three/tsl";
    import { ProgressRing } from "@skeletonlabs/skeleton-svelte";
    import { Segment } from "@skeletonlabs/skeleton-svelte";

    export let spotMetricsData;
    export let clusterColorScale;
    export let allLog;
    export let cellChat;
    export let baseApi;
    export let hoveredBarcode;
    export let hvg = {};
    export let availableClusters;
    export let umapData;
    export let clusterGeneExpression;
    export let clusterGeneDot;
    const GeneMode = ["Bar", "Sankey"];
    let currentGeneMode = "Bar";
    let hvging = false;

    const dispatch = createEventDispatcher();

    let violinDiv;
    let donutDiv;
    let umapDiv;
    let barChartDiv;
    let sankeyDiv;
    let cellChatDiv;
    let ligandReceptorDiv;
    let clusterGeneExpressionDiv;
    let clusterGeneDotDiv;

    let expandedIndex = null;
    let umap;
    let enrichmentResults;
    let currentCluster;

    let groups = ["Overview", "Cluster", "SVG", "Log", "Spatial Communication"];
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
        });
    }

    $: if (
        group === "Spatial Communication" &&
        cellChat &&
        cellChatDiv &&
        ligandReceptorDiv
    ) {
        tick().then(() => {
            // drawCellChat();
            drawClusterChat();
            drawLigandReceptorChat();
        });
    }

    $: if (group === "Overview" && umapDiv && umapData) {
        tick().then(() => {
            drawUMAP();
        });
    }

    $: if (hoveredBarcode.from === "spotPlot" && umapData && umapDiv) {
        drawUMAP();
    }

    $: if (
        group === "Cluster" &&
        spotMetricsData &&
        donutDiv &&
        clusterGeneExpressionDiv &&
        clusterGeneDot
    ) {
        tick().then(() => {
            drawClusterBar(spotMetricsData);
            loadHeatmap();
            drawDotPlot();
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

        observeResize(violinDiv, () => Plotly.Plots.resize(violinDiv));
    }

    // function drawDonut(data) {
    //     // 只用一次 cluster 数据，不用重复的 metric
    //     const clusterCounts = {};

    //     data.forEach((d) => {
    //         if (!clusterCounts[d.cluster]) clusterCounts[d.cluster] = 0;
    //         clusterCounts[d.cluster]++;
    //     });

    //     const clusters = Object.keys(clusterCounts).sort((a, b) => +a - +b);
    //     const values = clusters.map((c) => clusterCounts[c]);
    //     const labels = clusters.map((c) => ` Cluster ${+c}`); // +1 for display

    //     const colors = clusters.map((c) => clusterColorScale(c));
    //     const trace = {
    //         type: "pie",
    //         labels,
    //         values,
    //         hole: 0.5, // Donut hole
    //         marker: { colors },
    //         textinfo: "percent",
    //         hoverinfo: "label+value",
    //     };

    //     const layout = {
    //         title: "Spot Count by Cluster",
    //         margin: { l: 0, r: 0, t: 0, b: 0 },
    //         showlegend: false,
    //         autosize: true,
    //         width: donutDiv.clientWidth / 2,
    //     };

    //     Plotly.newPlot(donutDiv, [trace], layout, {
    //         responsive: true,
    //         useResizeHandler: true,
    //         displaylogo: false,
    //         modeBarButtons: [[]],
    //     });

    //     observeResize(donutDiv, () => Plotly.Plots.resize(donutDiv));

    //     window.addEventListener("resize", () => {
    //         Plotly.Plots.resize(donutDiv);
    //     });
    // }

    function drawClusterBar(data) {
        const clusterCounts = {};

        data.forEach((d) => {
            if (!clusterCounts[d.cluster]) clusterCounts[d.cluster] = 0;
            clusterCounts[d.cluster]++;
        });

        const clusters = Object.keys(clusterCounts).sort((a, b) => +a - +b);

        const traces = clusters.map((c) => ({
            type: "bar",
            name: `Cluster ${c}`,
            y: ["Clusters"],
            x: [clusterCounts[c]],
            orientation: "h",
            marker: { color: clusterColorScale(c) },
            hovertemplate: `Cluster ${c}<br>Count: ${clusterCounts[c]}<extra></extra>`,
        }));

        const layout = {
            barmode: "stack",
            title: "Spot Count by Cluster",
            margin: { l: 0, r: 0, t: 0, b: 0 },
            showlegend: false,
            height: 50,
            xaxis: { visible: false },
            yaxis: { visible: false },
        };

        Plotly.newPlot(donutDiv, traces, layout, {
            responsive: true,
            useResizeHandler: true,
            displaylogo: false,
            modeBarButtons: [[]],
        });

        observeResize(donutDiv, () => Plotly.Plots.resize(donutDiv));
        window.addEventListener("resize", () => {
            Plotly.Plots.resize(donutDiv);
        });
    }

    async function loadHeatmap() {
        const genes = clusterGeneExpression.genes;
        const barcodes = clusterGeneExpression.barcodes;
        const expression = clusterGeneExpression.expression;
        const clusters = clusterGeneExpression.clusters;

        // 1️⃣ 排序 barcodes / clusters / expression
        const indices = [...barcodes.keys()].sort((i, j) =>
            clusters[i].localeCompare(clusters[j]),
        );

        const sortedBarcodes = indices.map((i) => barcodes[i]);
        const sortedClusters = indices.map((i) => clusters[i]);
        const sortedExpression = expression.map((row) =>
            indices.map((i) => row[i]),
        );

        // 2️⃣ 计算 cluster 分段 span（用于 color bar）
        const clusterSpans = [];
        let prev = sortedClusters[0];
        let start = 0;
        for (let i = 1; i <= sortedClusters.length; i++) {
            if (sortedClusters[i] !== prev) {
                clusterSpans.push({ cluster: prev, start, end: i - 1 });
                prev = sortedClusters[i];
                start = i;
            }
        }

        // 3️⃣ 构造 heatmap trace
        const heatmap = {
            z: sortedExpression,
            x: sortedBarcodes,
            y: genes,
            type: "heatmap",
            colorscale: "RdBu",
            zmin: -2,
            zmax: 2,
            showscale: true,
            xgap: 0,
            ygap: 0,
        };

        // 4️⃣ 构造 cluster color bar 作为 layout.shapes
        const shapes = clusterSpans.map((span) => ({
            type: "rect",
            xref: "x",
            yref: "paper",
            x0: sortedBarcodes[span.start],
            x1: sortedBarcodes[span.end],
            y0: 1.01,
            y1: 1.05,
            fillcolor: clusterColorScale(span.cluster),
            line: { width: 0 },
        }));

        // 5️⃣ 设置 layout，去掉原来的 bar trace 和 yaxis2
        const layout = {
            title: "Auto-selected Marker Gene Heatmap",
            margin: { t: 60, l: 120 },
            height: 400 + genes.length * 10,
            xaxis: {
                tickangle: 90,
                side: "bottom",
                showticklabels: false,
            },
            yaxis: {
                autorange: "reversed",
            },
            shapes: shapes,
            showlegend: false,
        };

        Plotly.newPlot(clusterGeneExpressionDiv, [heatmap], layout, {
            responsive: true,
            displayModeBar: false,
        });

        observeResize(clusterGeneExpressionDiv, () =>
            Plotly.Plots.resize(clusterGeneExpressionDiv),
        );
    }

    function drawDotPlot() {
        const { genes, clusters, data } = clusterGeneDot;

        // 构建矩阵：avg_expr（颜色） & pct_expr（圆大小）
        const avgMatrix = clusters.map((cl) =>
            genes.map((g) => {
                const entry = data.find(
                    (d) => d.gene === g && d.cluster === cl,
                );
                return entry ? entry.avg_expr : 0;
            }),
        );

        const pctMatrix = clusters.map((cl) =>
            genes.map((g) => {
                const entry = data.find(
                    (d) => d.gene === g && d.cluster === cl,
                );
                return entry ? entry.pct_expr * 100 : 0;
            }),
        );

        const trace = {
            type: "scatter",
            mode: "markers",
            x: genes,
            y: clusters,
            marker: {
                size: pctMatrix.flat(),
                sizemode: "area",
                sizeref: (2.0 * Math.max(...pctMatrix.flat())) / 40 ** 2, // 控制最大圆
                sizemin: 2,
                color: avgMatrix.flat(),
                colorscale: "PuBuGn",
                colorbar: {
                    title: "Average Expression",
                },
            },
            hovertemplate:
                "<b>%{y}</b> – %{x}<br>" +
                "Avg Expr: %{marker.color:.2f}<br>" +
                "Pct Expr: %{marker.size:.1f}%<extra></extra>",
        };

        const layout = {
            title: "Dotplot: Gene Expression per Cluster",
            xaxis: {
                title: "Gene",
                tickangle: -45,
                automargin: true,
                autosize: true,
                width: clusterGeneDotDiv.clientWidth,
            },
            yaxis: {
                title: "Cluster",
                automargin: true,
                autorange: "reversed",
            },
            height: 60 * clusters.length + 100,
            margin: { t: 40, b: 80, l: 80, r: 40 },
        };

        Plotly.newPlot(clusterGeneDotDiv, [trace], layout, {
            responsive: true,
            useResizeHandler: true,
            displayModeBar: false,
        });

        observeResize(clusterGeneDotDiv, () =>
            Plotly.Plots.resize(clusterGeneDotDiv),
        );

        window.addEventListener("resize", () => {
            Plotly.Plots.resize(clusterGeneDotDiv);
        });
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

        observeResize(umapDiv, () => Plotly.Plots.resize(umapDiv));

        window.addEventListener("resize", () => {
            Plotly.Plots.resize(umapDiv);
        });
    }

    $: if (hvg && barChartDiv && group === "SVG" && currentGeneMode === "Bar") {
        tick().then(() => {
            // console.log(hvg);
            drawEnrichmentChart(hvg[currentCluster]);
        });
    }

    $: if (hvg && sankeyDiv && group === "SVG" && currentGeneMode !== "Bar") {
        tick().then(() => {
            // 参数：前10个term，每个term最多取前5个gene
            const topTermCount = 10;
            const topGenePerTerm = 5;

            let data = hvg[currentCluster];

            // Step 1: 按 Adjusted P-value 取前 topTermCount 个术语
            const topTerms = [...data]
                .sort((a, b) => a["Adjusted P-value"] - b["Adjusted P-value"])
                .slice(0, topTermCount);

            // Step 2: 构建新的 filtered 数据结构（每个 term 的前 N 个 gene）
            const filteredResults = topTerms.map((term) => {
                const geneList = term.Genes.split(";").map((g) => g.trim());
                const topGenes = geneList.slice(0, topGenePerTerm); // 取前 N 个基因
                return {
                    ...term,
                    Genes: topGenes.join(";"), // 重新组合回字符串
                };
            });

            // Step 3: 传入你的桑基图绘图函数
            drawSankey(filteredResults);
        });
    }

    function drawEnrichmentChart(results) {
        if (!Array.isArray(results)) {
            // console.warn("无效的富集结果：", results);
            return;
        }
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

        observeResize(barChartDiv, () => Plotly.Plots.resize(barChartDiv));

        window.addEventListener("resize", () => {
            Plotly.Plots.resize(barChartDiv);
        });
    }

    function drawSankey(results) {
        const geneSet = new Set();
        const termSet = [];
        const links = [];

        // 收集所有基因和Term
        results.forEach((item) => {
            const genes = item.Genes.split(";");
            const term = item.Term;
            termSet.push(term);
            genes.forEach((g) => {
                geneSet.add(g);
                links.push({ source: g, target: term });
            });
        });

        const genes = [...geneSet];
        const terms = [...new Set(termSet)];
        const nodes = genes.concat(terms);
        const nodeIndex = Object.fromEntries(nodes.map((n, i) => [n, i]));

        // 定义颜色比例尺

        const pastelColors = [
            "#aec6cf",
            "#ffb347",
            "#77dd77",
            "#f49ac2",
            "#cfcfc4",
            "#b39eb5",
            "#ff6961",
            "#cb99c9",
            "#fdfd96",
            "#836953",
        ];

        const lightColors = [
            "#b3cde0",
            "#decbe4",
            "#fed9a6",
            "#ccebc5",
            "#fbb4ae",
            "#e5d8bd",
            "#f2f2f2",
            "#d9d9d9",
            "#e6f5c9",
            "#fddaec",
        ];
        const geneColorScale = d3.scaleOrdinal(pastelColors);
        const termColorScale = d3.scaleOrdinal(lightColors);

        // 为每个节点生成颜色
        const nodeColors = nodes.map((n, i) =>
            i < genes.length ? geneColorScale(n) : termColorScale(n),
        );

        // 构建颜色映射表：node name -> color
        const nodeColorMap = Object.fromEntries(
            nodes.map((n, i) => [n, nodeColors[i]]),
        );

        // 每条连接线颜色 == source 的颜色
        const linkColors = links.map((l) => nodeColorMap[l.source]);

        function hexToRgb(hex) {
            hex = hex.replace("#", "");
            const bigint = parseInt(hex, 16);
            const r = (bigint >> 16) & 255;
            const g = (bigint >> 8) & 255;
            const b = bigint & 255;
            return [r, g, b];
        }

        const data = {
            type: "sankey",
            orientation: "h",
            node: {
                pad: 6,
                thickness: 20,
                line: { color: "black", width: 0.5 },
                label: nodes,
                color: nodeColors,
            },
            link: {
                source: links.map((l) => nodeIndex[l.source]),
                target: links.map((l) => nodeIndex[l.target]),
                value: links.map(() => 1),
                color: links.map((l) => {
                    const hex = geneColorScale(l.source);
                    const [r, g, b] = hexToRgb(hex);
                    return `rgba(${r}, ${g}, ${b}, 0.5)`; // 半透明
                }),
            },
        };

        const layout = {
            title: "Sankey: Genes → Enriched Terms",
            font: { size: 10 },
            margin: { l: 20, r: 20, t: 40, b: 10 },
            height: Math.max(400, nodes.length * 20), // 自适应高度
        };

        Plotly.newPlot(sankeyDiv, [data], layout, {
            displaylogo: false,
            responsive: true,
        });

        observeResize(sankeyDiv, () => Plotly.Plots.resize(sankeyDiv));

        window.addEventListener("resize", () => {
            Plotly.Plots.resize(sankeyDiv);
        });
    }

    async function getHvg() {
        hvging = true;
        const hvgRes = await fetch(
            baseApi + `/hvg-enrichment-cluster?cluster=${currentCluster}`,
        );
        const hvgData = await hvgRes.json();
        console.log(hvgData);
        // 提取 hvgData 对象中唯一的 key（如 "1.0"）
        const rawClusterKey = Object.keys(hvgData)[0];

        // 获取该 key 对应的富集结果数组
        const enrichmentResults = hvgData[rawClusterKey];

        // 将它存入 hvg 对象，用 currentCluster 作为 key
        hvg = {
            ...hvg,
            [currentCluster]: enrichmentResults,
        };
        console.log(hvg);
        hvging = false;
    }

    function observeResize(dom, callback) {
        if (!dom) return;
        const ro = new ResizeObserver(() => {
            callback?.();
        });
        ro.observe(dom);
        return () => ro.disconnect();
    }

    async function drawCellChat() {
        const nodes = [];
        const nodeMap = new Map(); // label -> index
        const links = [];

        function getNodeIndex(label) {
            if (!nodeMap.has(label)) {
                nodeMap.set(label, nodes.length);
                nodes.push(label);
            }
            return nodeMap.get(label);
        }

        // 强制将 source_cluster 和 target_cluster 分开处理
        const sourceClusters = new Set();
        const targetClusters = new Set();
        const ligands = new Set();
        const receptors = new Set();

        // 先收集节点
        cellChat.forEach((row) => {
            sourceClusters.add(`SRC:${row.source_cluster}`);
            targetClusters.add(`TGT:${row.target_cluster}`);
            ligands.add(`L:${row.ligand}`);
            receptors.add(`R:${row.receptor}`);
        });

        // 按顺序加入节点：Source Cluster → Ligand → Receptor → Target Cluster
        [
            ...sourceClusters,
            ...ligands,
            ...receptors,
            ...targetClusters,
        ].forEach((label) => {
            getNodeIndex(label);
        });

        // 创建链接
        cellChat.forEach((row) => {
            const sourceCluster = `SRC:${row.source_cluster}`;
            const ligand = `L:${row.ligand}`;
            const receptor = `R:${row.receptor}`;
            const targetCluster = `TGT:${row.target_cluster}`;
            const score = row.interaction_score;

            links.push({
                source: nodeMap.get(sourceCluster),
                target: nodeMap.get(ligand),
                value: score,
            });
            links.push({
                source: nodeMap.get(ligand),
                target: nodeMap.get(receptor),
                value: score,
            });
            links.push({
                source: nodeMap.get(receptor),
                target: nodeMap.get(targetCluster),
                value: score,
            });
        });

        const labelClean = (label) => {
            return label
                .replace(/^SRC:/, "Cluster ")
                .replace(/^TGT:/, "Cluster ")
                .replace(/^L:/, "")
                .replace(/^R:/, "");
        };

        const colorMap = (label) => {
            if (label.startsWith("SRC:")) return "#636efa";
            if (label.startsWith("TGT:")) return "#636efa";
            if (label.startsWith("L:")) return "#EF553B";
            if (label.startsWith("R:")) return "#00cc96";
            return "#AB63FA";
        };

        const sankeyData = {
            type: "sankey",
            orientation: "h",
            node: {
                pad: 15,
                thickness: 20,
                line: { color: "black", width: 0.5 },
                label: nodes.map(labelClean),
                color: nodes.map(colorMap),
            },
            link: {
                source: links.map((d) => d.source),
                target: links.map((d) => d.target),
                value: links.map((d) => d.value),
            },
        };

        const layout = {
            title: "Cluster → Ligand → Receptor → Cluster Communication",
            font: { size: 12 },
            height: 900,
            width: 1300,
        };

        Plotly.newPlot(cellChatDiv, [sankeyData], layout);
    }

    function drawClusterChat() {
        console.log(cellChat);
        d3.select(cellChatDiv).select("svg").remove();
        const width = 500;
        const height = width;
        const innerRadius = Math.min(width, height) * 0.5 - 20;
        const outerRadius = innerRadius + 6;

        const data = cellChat.map((d) => ({
            source: String(d.source_cluster),
            target: String(d.target_cluster),
            value: d.interaction_score,
        }));

        const names = Array.from(
            new Set(data.flatMap((d) => [d.source, d.target])),
        );
        const index = new Map(names.map((name, i) => [name, i]));

        const matrix = Array.from({ length: names.length }, () =>
            new Array(names.length).fill(0),
        );

        for (const { source, target, value } of data) {
            matrix[index.get(source)][index.get(target)] += value;
        }

        const chord = d3
            .chordDirected()
            .padAngle(12 / innerRadius)
            .sortSubgroups(d3.descending)
            .sortChords(d3.descending);

        const chords = chord(matrix);
        chords.groups.forEach((d) => {
            d.name = names[d.index];
        });

        chords.forEach((d) => {
            d.source.name = names[d.source.index];
            d.target.name = names[d.target.index];
        });
        // console.log(chords);

        // 每个 cluster 的交互总量（outgoing + incoming）
        const groupScores = chords.groups.map((d) => {
            const out = d3.sum(matrix[d.index]);
            const _in = d3.sum(matrix, (row) => row[d.index]);
            return out + _in;
        });

        const radiusScale = d3
            .scaleSqrt()
            .domain([0, d3.max(groupScores)])
            .range([innerRadius, innerRadius + 15]);

        // const arc = d3.arc().innerRadius(innerRadius).outerRadius(outerRadius);
        const arc = d3
            .arc()
            .innerRadius(innerRadius)
            .outerRadius((d) => radiusScale(groupScores[d.index]));

        const ribbon = d3
            .ribbonArrow()
            .radius(innerRadius - 0.5)
            .padAngle(1 / innerRadius);

        const colors = d3.schemeCategory10;

        const formatValue = (x) => x.toFixed(6);

        const svg = d3
            .select(cellChatDiv)
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", [-width / 2, -height / 2, width, height])
            .attr("style", "width: 100%; height: auto; font: 10px sans-serif;");

        const textId = "text-path-id";

        svg.append("path")
            .attr("id", textId)
            .attr("fill", "none")
            .attr(
                "d",
                d3.arc()({ outerRadius, startAngle: 0, endAngle: 2 * Math.PI }),
            );

        // Ribbons (arrows)
        svg.append("g")
            .attr("fill-opacity", 0.75)
            .selectAll("path")
            .data(chords)
            .join("path")
            .attr("d", ribbon)
            .attr("fill", (d) => clusterColorScale(d.source.name))
            .style("mix-blend-mode", "multiply")
            .append("title")
            .text((d) => {
                const sourceName = names[d.source.index];
                const targetName = names[d.target.index];
                const matched = data.find(
                    (item) =>
                        item.source === sourceName &&
                        item.target === targetName,
                );
                const score = matched ? matched.value : 0;
                return `${sourceName} → ${targetName}\nScore: ${formatValue(score)}`;
            });

        // Arcs and labels
        const g = svg.append("g").selectAll("g").data(chords.groups).join("g");

        g.append("path")
            .attr("d", arc)
            .attr("fill", (d) => clusterColorScale(d.name))
            .attr("stroke", "#fff");

        g.append("text")
            .attr("dy", -3)
            .append("textPath")
            .attr("xlink:href", `#${textId}`)
            .attr("startOffset", (d) => d.startAngle * outerRadius)
            .text((d) => names[d.index]);

        g.append("title").text((d) => {
            const out = d3.sum(matrix[d.index]);
            const _in = d3.sum(matrix, (row) => row[d.index]);
            return `${names[d.index]}
→ others: ${formatValue(out)}
← from others: ${formatValue(_in)}`;
        });
    }

    function drawLigandReceptorChat() {
        console.log(cellChat);
        d3.select(ligandReceptorDiv).select("svg").remove();
        const width = 500;
        const height = width;
        const innerRadius = Math.min(width, height) * 0.5 - 20;
        const outerRadius = innerRadius + 6;

        const data = cellChat.map((d) => ({
            source: String(d.ligand),
            target: String(d.receptor),
            value: d.interaction_score,
        }));
        console.log(data);
        const names = Array.from(
            new Set(data.flatMap((d) => [d.source, d.target])),
        );

        const index = new Map(names.map((name, i) => [name, i]));

        const matrix = Array.from({ length: names.length }, () =>
            new Array(names.length).fill(0),
        );

        for (const { source, target, value } of data) {
            matrix[index.get(source)][index.get(target)] += value;
        }

        const chord = d3
            .chordDirected()
            .padAngle(12 / innerRadius)
            .sortSubgroups(d3.descending)
            .sortChords(d3.descending);

        const chords = chord(matrix);
        chords.groups.forEach((d) => {
            d.name = names[d.index];
        });

        chords.forEach((d) => {
            d.source.name = names[d.source.index];
            d.target.name = names[d.target.index];
        });
        console.log(chords);

        // 每个 cluster 的交互总量（outgoing + incoming）
        const groupScores = chords.groups.map((d) => {
            const out = d3.sum(matrix[d.index]);
            const _in = d3.sum(matrix, (row) => row[d.index]);
            return out + _in;
        });

        // 构建一个 scale 来控制圆的大小
        const radiusScale = d3
            .scaleSqrt()
            .domain([0, d3.max(groupScores)])
            .range([innerRadius, innerRadius + 15]); // 控制圆的大小变化幅度

        // const arc = d3.arc().innerRadius(innerRadius).outerRadius(outerRadius);
        const arc = d3
            .arc()
            .innerRadius(innerRadius)
            .outerRadius((d) => radiusScale(groupScores[d.index]));

        const ribbon = d3
            .ribbonArrow()
            .radius(innerRadius - 0.5)
            .padAngle(1 / innerRadius);

        const colors = d3.schemeCategory10;

        const formatValue = (x) => x.toFixed(6);

        const svg = d3
            .select(ligandReceptorDiv)
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", [-width / 2, -height / 2, width, height])
            .attr("style", "width: 100%; height: auto; font: 10px sans-serif;");

        const textId = "text-path-id";

        svg.append("path")
            .attr("id", textId)
            .attr("fill", "none")
            .attr(
                "d",
                d3.arc()({ outerRadius, startAngle: 0, endAngle: 2 * Math.PI }),
            );

        // Ribbons (arrows)
        svg.append("g")
            .attr("fill-opacity", 0.75)
            .selectAll("path")
            .data(chords)
            .join("path")
            .attr("d", ribbon)
            .attr("fill", (d) => clusterColorScale(d.source.name))
            .style("mix-blend-mode", "multiply")
            .append("title")
            .text((d) => {
                const sourceName = names[d.source.index];
                const targetName = names[d.target.index];
                const matched = data.find(
                    (item) =>
                        item.source === sourceName &&
                        item.target === targetName,
                );
                const score = matched ? matched.value : 0;
                return `${sourceName} → ${targetName}\nScore: ${formatValue(score)}`;
            });

        // Arcs and labels
        const g = svg.append("g").selectAll("g").data(chords.groups).join("g");

        g.append("path")
            .attr("d", arc)
            .attr("fill", (d) => clusterColorScale(d.name))
            .attr("stroke", "#fff");

        g.append("text")
            .attr("dy", -3)
            .append("textPath")
            .attr("xlink:href", `#${textId}`)
            .attr("startOffset", (d) => d.startAngle * outerRadius)
            .text((d) => names[d.index]);

        g.append("title").text((d) => {
            const out = d3.sum(matrix[d.index]);
            const _in = d3.sum(matrix, (row) => row[d.index]);
            return `${names[d.index]}
→ others: ${formatValue(out)}
← from others: ${formatValue(_in)}`;
        });
    }

    onMount(() => {
        if (availableClusters) currentCluster = availableClusters[0];
    });
</script>

{#if hvging}
    <div
        class="fixed inset-0 z-50 flex justify-center items-center bg-white/80"
    >
        <ProgressRing
            value={null}
            size="size-14"
            meterStroke="stroke-blue-300"
            trackStroke="stroke-blue-400"
        />
    </div>
{/if}

<Tabs
    bind:value={group}
    onValueChange={(e) => (group = e.value)}
    class="h-full max-h-full overflow-y-scroll"
>
    {#snippet list()}
        <div>
            {#each groups as g}
                <Tabs.Control value={g}>{g}</Tabs.Control>
            {/each}
        </div>
    {/snippet}

    {#snippet content()}
        <div class="overflow-y-scroll scrollbar-none max-h-full h-full">
            {#each groups as g}
                <Tabs.Panel value={g} class="max-h-full h-full">
                    <div class="h-full overflow-y-scroll max-h-full">
                        {#if g === "Overview"}
                            <div bind:this={umapDiv}></div>
                            <div bind:this={violinDiv}></div>
                        {:else if g === "Cluster"}
                            <div>Count Distribution</div>
                            <div
                                bind:this={donutDiv}
                                class="w-full max-w-full"
                            ></div>
                            <div
                                bind:this={clusterGeneExpressionDiv}
                                class="w-full h-full"
                            ></div>
                            <div
                                bind:this={clusterGeneDotDiv}
                                class="w-full h-full"
                            ></div>
                        {:else if g === "Log" && allLog}
                            <div class="table-wrap">
                                <table
                                    class="table caption-bottom text-xs w-full"
                                >
                                    <thead>
                                        <tr>
                                            <th>Barcode</th>
                                            <th>Prev</th>
                                            <th>New</th>
                                        </tr>
                                    </thead>
                                    <tbody
                                        class="[&>tr]:hover:preset-tonal-primary"
                                    >
                                        {#each allLog as row, i}
                                            <tr
                                                class="cursor-pointer"
                                                on:click={() =>
                                                    (expandedIndex =
                                                        expandedIndex === i
                                                            ? null
                                                            : i)}
                                            >
                                                <td>{row.barcode}</td>
                                                <td>{row.old_cluster}</td>
                                                <td>{row.new_cluster}</td>
                                            </tr>
                                            {#if expandedIndex === i}
                                                <tr class="bg-muted/30 text-sm">
                                                    <td colspan="3">
                                                        <div>
                                                            <strong
                                                                >Comment:</strong
                                                            >
                                                            {row.comment || "-"}
                                                        </div>
                                                        <div>
                                                            <strong
                                                                >Time:</strong
                                                            >
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
                                            <td class="text-right"
                                                >{allLog.length}</td
                                            >
                                        </tr>
                                    </tfoot>
                                </table>
                            </div>
                        {:else if g === "SVG" && hvg}
                            <div
                                class="w-full flex flex-col items-center space-y-4"
                            >
                                <div
                                    class="w-full flex flex-row gap-6 items-end"
                                >
                                    <!-- Cluster 选择器 -->
                                    <div class="w-64 flex flex-col space-y-1">
                                        <label
                                            class="text-sm font-medium text-gray-700"
                                            >Cluster</label
                                        >
                                        <select
                                            class="w-full border border-gray-300 rounded px-3 py-2 bg-white focus:ring-2 focus:ring-stone-400"
                                            bind:value={currentCluster}
                                        >
                                            {#each availableClusters as c}
                                                <option value={c}>{c}</option>
                                            {/each}
                                        </select>
                                    </div>

                                    <!-- Gene Mode Segment Control -->
                                    <div class="flex flex-col space-y-1">
                                        {#if !hvg[currentCluster]}
                                            <button
                                                type="button"
                                                class="btn preset-filled"
                                                on:click={getHvg}
                                                >Analyse</button
                                            >
                                        {:else}
                                            <label
                                                class="text-sm font-medium text-gray-700"
                                                >Chart Type</label
                                            >
                                            <Segment
                                                name="size"
                                                value={currentGeneMode}
                                                onValueChange={(e) =>
                                                    (currentGeneMode = e.value)}
                                                class="w-full flex"
                                            >
                                                {#each GeneMode as gm}
                                                    <Segment.Item value={gm}
                                                        >{gm}</Segment.Item
                                                    >
                                                {/each}
                                            </Segment>
                                        {/if}
                                    </div>
                                </div>

                                <!-- Chart Display Area -->
                                <div class="w-full max-w-5xl">
                                    {#if hvg[currentCluster]}
                                        {#if currentGeneMode === "Bar"}
                                            <div
                                                bind:this={barChartDiv}
                                                class="w-full h-full"
                                            ></div>
                                        {:else}
                                            <div
                                                bind:this={sankeyDiv}
                                                class="w-full h-full"
                                            ></div>
                                        {/if}
                                    {/if}
                                </div>
                            </div>
                        {:else if g === "Spatial Communication" && cellChat}
                            <div>cluster</div>
                            <div
                                bind:this={cellChatDiv}
                                class="w-full max-w-full"
                            ></div>
                            <div>Ligand and Receptor</div>
                            <div
                                bind:this={ligandReceptorDiv}
                                class="w-full max-w-full"
                            ></div>
                        {/if}
                    </div>
                </Tabs.Panel>
            {/each}
        </div>
    {/snippet}
</Tabs>

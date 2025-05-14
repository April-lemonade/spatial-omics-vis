<script>
    import { onMount } from "svelte";
    import Plotly from "plotly.js-dist-min";
    import { Edit } from "@lucide/svelte";
    import { Tabs } from "@skeletonlabs/skeleton-svelte";
    import * as d3 from "d3";
    import Plot from "./component/plot.svelte";
    import Overview from "./component/overview.svelte";
    import SpotInspection from "./component/spotInspection.svelte";

    const baseApi = "http://localhost:8000";
    const imageUrl = "http://localhost:8000/images/tissue_hires_image.png";

    let spatialDiv, heatmapDiv;
    let clickedInfo;
    let spatialData;
    let spatialInfo;

    let availableClusters;

    let allSlices, currentSlice;
    let ncountSpatialData, spotMetricsData;
    let clusterColorScale;
    let allLog;

    async function fetchSpatial() {
        // 先获取所有的切片 ID
        const slicesRes = await fetch(baseApi + "/allslices");
        allSlices = await slicesRes.json();
        currentSlice = allSlices[0];

        // 用当前切片 ID 获取 plot-data 和 slice-info
        const [plotRes, infoRes, ncountRes, metricsRes, logRes] =
            await Promise.all([
                fetch(`${baseApi}/plot-data?slice_id=${currentSlice}`),
                fetch(`${baseApi}/slice-info?slice_id=${currentSlice}`),
                fetch(`${baseApi}/ncount_by_cluster?slice_id=${currentSlice}`),
                fetch(`${baseApi}/spot-metrics?slice_id=${currentSlice}`),
                fetch(`${baseApi}/cluster-log?slice_id=${currentSlice}`),
            ]);

        const plotData = await plotRes.json();
        const sliceInfo = await infoRes.json();
        const ncountData = await ncountRes.json();
        const metricsData = await metricsRes.json();
        const logData = await logRes.json();
        // console.log(metricsData[0]);

        return { plotData, sliceInfo, ncountData, metricsData, logData };
    }

    async function drawExpressionMatrix() {
        const res = await fetch(`${baseApi}/expression-matrix`);
        const matrixData = await res.json(); // [{gene1: val1, gene2: val2, ...}, ...]

        const geneList = Object.keys(matrixData[0]);
        const barcodes = matrixData.map((_, i) => `spot ${i}`);
        const values = matrixData.map((row) => geneList.map((g) => row[g]));

        const trace = {
            z: values,
            x: geneList,
            y: barcodes,
            type: "heatmap",
            colorscale: "YlGnBu",
        };

        Plotly.newPlot("heatmapDiv", [trace], {
            title: "Gene Expression Heatmap (All Spots)",
            height: 800,
            xaxis: { title: "Gene" },
            yaxis: { title: "Spot (Barcode)", automargin: true },
        });
    }

    function handleSpotClick(info) {
        // console.log("选中了一个 spot:", info.barcode, info);
        clickedInfo = info;
    }

    async function handleClusterUpdate({
        barcode,
        newCluster,
        oldCluster,
        comment,
    }) {
        const res = await fetch(`${baseApi}/update-cluster`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                slice_id: currentSlice,
                barcode,
                old_cluster: oldCluster,
                new_cluster: newCluster,
                comment,
            }),
        });

        if (res.ok) {
            // clusterEdit = false;

            // ✅ 成功更新后重新获取并刷新 spatialData
            const updated = await fetch(
                `${baseApi}/plot-data?slice_id=${currentSlice}`,
            );
            spatialData = await updated.json();
            clickedInfo.cluster = newCluster;
            // ✅ 重新绘图（如果是子组件，可以用 event 触发父组件刷新）
            // Plotly.react(spatialDiv, spatialData, plotInstance.layout);
        }

        // // ✅ 更新 spatialData 中的点
        // moveBarcodeToCluster(barcode, oldCluster, newCluster);
        // Plotly.react(spatialDiv, spatialData, plotInstance.layout);
        console.log({
            barcode,
            newCluster,
            oldCluster,
            comment,
        });
    }

    onMount(async () => {
        const image = new Image();
        image.src = imageUrl;
        await new Promise((resolve) => (image.onload = resolve));
        const { ncountData, plotData, sliceInfo, metricsData, logData } =
            await fetchSpatial();
        spatialData = plotData;
        spatialInfo = sliceInfo;
        ncountSpatialData = ncountData;
        spotMetricsData = metricsData;
        allLog = logData;
        // console.log(spotMetricsData[0]);

        const clusters = new Set();
        plotData.forEach((trace) => {
            clusters.add(trace.name);
        });
        availableClusters = Array.from(clusters);
        clusterColorScale = d3
            .scaleOrdinal(d3.schemeTableau10)
            .domain(availableClusters);
    });
</script>

<div class="grid h-screen grid-rows-[auto_1fr_auto] gap-y-2">
    <!-- Header -->
    <header class="text-xl p-3 bg-gray-200">空转数据可视化demo</header>
    <!-- Grid Column -->
    <div class="grid grid-cols-1 md:grid-cols-[15%_50%_34%] px-1 gap-x-1">
        <!-- Sidebar (Left) -->
        <aside class="p-4 border-1 border-solid rounded-lg border-stone-300">
            <span class="font-bold">slice:</span>
            <select class="select" bind:value={currentSlice}>
                {#each allSlices as slice}
                    <option value={slice}>{slice}</option>
                {/each}
            </select>
            {#if spatialInfo}
                {#each Object.entries(spatialInfo) as [key, value]}
                    {#if key !== "expression"}
                        <div class="mb-2">
                            <strong>{key}:</strong>
                            <div class="flex flex-row items-center gap-2">
                                <span>{value}</span>
                            </div>
                        </div>
                    {/if}
                {/each}
            {/if}
        </aside>
        <!-- Main -->
        <main
            class=" p-1 space-y-4 h-full w-full flex flex-col border-1 border-solid rounded-lg border-stone-300"
        >
            <Plot
                {spatialData}
                {imageUrl}
                {clusterColorScale}
                on:spotClick={(e) => handleSpotClick(e.detail)}
            ></Plot>
        </main>
        <!-- Sidebar (Right) -->
        <aside
            class="p-4 border-1 border-solid rounded-lg border-stone-300"
            style="font-family: sans-serif;"
        >
            <!-- <header class="text-xl">Inspection View</header> -->
            <div>
                {#if clickedInfo}
                    <SpotInspection
                        {clickedInfo}
                        {availableClusters}
                        {baseApi}
                        {currentSlice}
                        on:clusterUpdate={(e) => handleClusterUpdate(e.detail)}
                    ></SpotInspection>
                {:else if spatialInfo}
                    <Overview {spotMetricsData} {clusterColorScale} {allLog}
                    ></Overview>
                {/if}
            </div>
        </aside>
    </div>
    <!-- Footer -->
    <footer class=" text-center">@2025.5</footer>
</div>

<style>
</style>

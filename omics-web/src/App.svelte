<script>
    import { onMount } from "svelte";
    import Plotly from "plotly.js-dist-min";
    import { Edit } from "@lucide/svelte";
    import { Tabs } from "@skeletonlabs/skeleton-svelte";
    import * as d3 from "d3";
    import Plot from "./component/plot.svelte";
    import Overview from "./component/overview.svelte";
    import SpotInspection from "./component/spotInspection.svelte";
    // import { RoomEnvironment } from "three/esxamples/jsm/Addons.js";
    import { ProgressRing } from "@skeletonlabs/skeleton-svelte";
    import Lassomode from "./component/lassomode.svelte";

    const baseApi = "http://localhost:8000";
    let imageUrl;
    const clusteringMethods = ["leiden"];

    let currentMethod = clusteringMethods[0];
    let spatialDiv, heatmapDiv;
    let clickedInfo;
    let spatialData;
    let spatialInfo;
    let expandedIndex = null;

    let availableClusters;

    let allSlices, currentSlice;
    let ncountSpatialData, spotMetricsData;
    let clusterColorScale;
    let hvg;
    let allLog;
    let lassoSelected = false;
    let reclusering = false;
    let reclustered = false;
    let hoveredBarcode = { barcode: "", from: "" };

    async function fetchSpatial() {
        // 先获取所有的切片 ID
        const slicesRes = await fetch(baseApi + "/allslices");
        allSlices = await slicesRes.json();
        currentSlice = allSlices[0];

        imageUrl = `${baseApi}/images/${currentSlice}/tissue_hires_image.png`;

        const image = new Image();
        image.src = imageUrl;
        await new Promise((resolve) => (image.onload = resolve));

        // 用当前切片 ID 获取 plot-data 和 slice-info
        const [plotRes, infoRes, ncountRes, metricsRes, logRes, hvgRes] =
            await Promise.all([
                fetch(`${baseApi}/plot-data?slice_id=${currentSlice}`),
                fetch(`${baseApi}/slice-info?slice_id=${currentSlice}`),
                fetch(`${baseApi}/ncount_by_cluster?slice_id=${currentSlice}`),
                fetch(`${baseApi}/spot-metrics?slice_id=${currentSlice}`),
                fetch(`${baseApi}/cluster-log?slice_id=${currentSlice}`),
                fetch(`${baseApi}/hvg-enrichment`),
            ]);

        const plotData = await plotRes.json();
        const sliceInfo = await infoRes.json();
        const ncountData = await ncountRes.json();
        const metricsData = await metricsRes.json();
        const logData = await logRes.json();
        const hvgData = await hvgRes.json();
        // console.log(metricsData[0]);

        return {
            plotData,
            sliceInfo,
            ncountData,
            metricsData,
            logData,
            hvgData,
        };
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

    function handleSpotClick(detail) {
        // console.log(detail);
        // console.log("选中了一个 spot:", info.barcode, info);
        clickedInfo = detail.info;
        lassoSelected = detail.lassoSelected;
        console.log(clickedInfo);
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
            // ✅ 成功更新后重新获取并刷新 spatialData
            const [updatedPlotRes, updatedLogRes] = await Promise.all([
                fetch(`${baseApi}/plot-data?slice_id=${currentSlice}`),
                fetch(`${baseApi}/cluster-log?slice_id=${currentSlice}`),
            ]);

            spatialData = await updatedPlotRes.json();
            allLog = await updatedLogRes.json();

            // ✅ 更新当前点击的点的聚类
            clickedInfo.cluster = newCluster;
            if (lassoSelected) {
                lassoSelected = false;
                clickedInfo = null;
            }
        }

        //  spatialData 中的点
        // console.log({
        //     barcode,
        //     newCluster,
        //     oldCluster,
        //     comment,
        // });
    }

    async function recluster() {
        reclusering = true;
        reclustered = false;
        const res = await fetch(`${baseApi}/recluster`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                slice_id: currentSlice,
                barcode: clickedInfo,
            }),
        });

        if (res.ok) {
            const data = await res.json();
            console.log("返回的数据内容：", data);
            clickedInfo = data;
            // lassoSelected = false;
            reclustered = true;
            // dispatch("spotClick", {
            //     info: data,
            //     lassoSelected: true,
            // });
            reclusering = false;
        }
    }

    onMount(async () => {
        const {
            ncountData,
            plotData,
            sliceInfo,
            metricsData,
            logData,
            hvgData,
        } = await fetchSpatial();
        spatialData = plotData;
        spatialInfo = sliceInfo;
        ncountSpatialData = ncountData;
        spotMetricsData = metricsData;
        allLog = logData;
        hvg = hvgData;
        // console.log(spotMetricsData[0]);

        const clusters = new Set();
        plotData.forEach((trace) => {
            clusters.add(trace.name);
        });
        availableClusters = Array.from(clusters);
        const clusterNames = availableClusters.sort((a, b) => {
            return +a.replace("Cluster ", "") - +b.replace("Cluster ", "");
        });
        clusterColorScale = d3
            .scaleOrdinal()
            .domain(clusterNames)
            .range(d3.schemeTableau10);
        // clusterColorScale = d3
        //     .scaleOrdinal(d3.schemeTableau10)
        //     .domain(availableClusters);
    });
</script>

{#if !spatialData}
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

<div class="grid h-screen grid-rows-[auto_1fr_auto] gap-y-2">
    <!-- Header -->
    <header class="text-xl p-3 bg-gray-200">空转数据可视化demo</header>
    <!-- Grid Column -->
    <div
        class="grid grid-cols-1 md:grid-cols-[17%_50%_32%] px-1 gap-x-1 h-full overflow-hidden
"
    >
        <!-- Sidebar (Left) -->
        <aside
            class="p-4 border border-stone-300 rounded-lg text-sm space-y-3 leading-relaxed"
        >
            <!-- Slice Selector -->
            <div>
                <div class="font-semibold mb-1 text-gray-700">Slice</div>
                <select
                    class="w-full border border-gray-300 rounded px-3 py-1 bg-white focus:outline-none focus:ring-1 focus:ring-stone-400"
                    bind:value={currentSlice}
                >
                    {#each allSlices as slice}
                        <option value={slice}>{slice}</option>
                    {/each}
                </select>
            </div>

            <!-- Clustering Method -->
            <div>
                <div class="font-semibold mb-1 text-gray-700">
                    Clustering Method
                </div>
                <select
                    class="w-full border border-gray-300 rounded px-3 py-1 bg-white focus:outline-none focus:ring-1 focus:ring-stone-400"
                    bind:value={currentMethod}
                >
                    {#each clusteringMethods as method}
                        <option value={method}>{method}</option>
                    {/each}
                </select>
            </div>

            <!-- Spatial Info -->
            {#if spatialInfo}
                <div
                    class="pt-2 border-t border-dashed border-stone-300 text-gray-600"
                >
                    {#each Object.entries(spatialInfo) as [key, value]}
                        {#if key !== "expression"}
                            <div class="flex justify-between text-sm py-0.5">
                                <span class="capitalize">{key}:</span>
                                <span class="text-gray-800">{value}</span>
                            </div>
                        {/if}
                    {/each}
                </div>
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
                {lassoSelected}
                {hoveredBarcode}
                on:spotClick={(e) => handleSpotClick(e.detail)}
                on:hover={(e) => {
                    hoveredBarcode = {
                        barcode: e.detail.barcode,
                        from: e.detail.from,
                    };
                    // console.log(hoveredBarcode);
                }}
            ></Plot>
        </main>
        <!-- Sidebar (Right) -->
        <aside
            class="p-4 border-1 border-solid rounded-lg border-stone-300 h-full overflow-y-scroll scrollbar-none"
            style="font-family: sans-serif;scrollbar-width: auto; scrollbar-color: #999 transparent;"
        >
            <!-- <header class="text-xl">Inspection View</header> -->
            <div class="h-full">
                {#if lassoSelected}
                    <!-- {#if reclustered && !reclusering}
                        <Lassomode {clickedInfo}></Lassomode>
                    {:else if reclusering}
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
                    {:else if clickedInfo}
                        <div>{clickedInfo?.length} spots selected</div>
                        <button
                            type="button"
                            class="btn preset-filled"
                            on:click={() => {
                                recluster();
                            }}>Recluster</button
                        >
                    {/if} -->
                    <div class="h-full">
                        <Lassomode
                            {clickedInfo}
                            {baseApi}
                            {currentSlice}
                            on:acceptRecluster={(e) =>
                                handleClusterUpdate(e.detail)}
                        ></Lassomode>
                    </div>
                {:else if clickedInfo}
                    <!-- {#if clickedInfo} -->
                    <SpotInspection
                        {clickedInfo}
                        {availableClusters}
                        {baseApi}
                        {currentSlice}
                        on:clusterUpdate={(e) => handleClusterUpdate(e.detail)}
                    ></SpotInspection>
                {:else if spatialInfo}
                    <Overview
                        {spotMetricsData}
                        {clusterColorScale}
                        {allLog}
                        {currentSlice}
                        {baseApi}
                        {hoveredBarcode}
                        {hvg}
                        on:hover={(e) => {
                            hoveredBarcode = {
                                barcode: e.detail.barcode,
                                from: e.detail.from,
                            };
                            console.log(hoveredBarcode);
                        }}
                    ></Overview>
                {/if}
            </div>
        </aside>
    </div>
    <!-- Footer -->
    <footer class=" text-center">@2025.5</footer>
</div>

<style>
    aside::-webkit-scrollbar {
        width: 8px;
    }
    aside::-webkit-scrollbar-track {
        background: transparent;
    }
    aside::-webkit-scrollbar-thumb {
        background-color: rgba(100, 100, 100, 0.4);
        border-radius: 4px;
    }
</style>

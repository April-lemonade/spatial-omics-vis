<script>
    import { onMount } from "svelte";
    import Split from "split.js";
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
    const clusteringMethods = ["mclust"];

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
    let epoch = 500;
    let n_clusters = 7;
    let umapData;
    let cellChat;
    let lassoHover;

    async function fecthUmapData() {
        const response = await fetch(
            baseApi + `/umap-coordinates?slice_id=${currentSlice}`,
        );
        const data = await response.json();
        return data;
    }

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
        const [plotRes, infoRes, ncountRes, metricsRes, logRes, cellChatRes] =
            await Promise.all([
                fetch(`${baseApi}/plot-data?slice_id=${currentSlice}`),
                fetch(`${baseApi}/slice-info?slice_id=${currentSlice}`),
                fetch(`${baseApi}/ncount_by_cluster?slice_id=${currentSlice}`),
                fetch(`${baseApi}/spot-metrics?slice_id=${currentSlice}`),
                fetch(`${baseApi}/cluster-log?slice_id=${currentSlice}`),
                fetch(`${baseApi}/cellchat`),
            ]);

        const plotData = await plotRes.json();
        const sliceInfo = await infoRes.json();
        const ncountData = await ncountRes.json();
        const metricsData = await metricsRes.json();
        const logData = await logRes.json();
        const cellChatData = await cellChatRes.json();

        if (sliceInfo.cluster_method !== "not_clustered") {
            umapData = await fecthUmapData();
        }

        return {
            plotData,
            sliceInfo,
            ncountData,
            metricsData,
            logData,
            cellChatData,
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

    async function handleClusterUpdate(info) {
        console.log(
            info.barcode,
            info.newCluster,
            info.oldCluster,
            info.comment,
        );
        const res = await fetch(`${baseApi}/update-cluster`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                slice_id: currentSlice,
                barcode: info.barcode,
                old_cluster: info.oldCluster,
                new_cluster: info.newCluster,
                comment: info.comment,
            }),
        });

        if (res.ok) {
            // ✅ 成功更新后重新获取并刷新 spatialData
            const [updatedPlotRes, updatedLogRes] = await Promise.all([
                fetch(`${baseApi}/plot-data?slice_id=${currentSlice}`),
                fetch(`${baseApi}/cluster-log?slice_id=${currentSlice}`),
            ]);

            // ✅ 更新当前点击的点的聚类
            clickedInfo.cluster = info.newCluster;
            // clickedInfo.changed = false;

            if (lassoSelected) {
                if (info.remainingChangedCount === 0) {
                    lassoSelected = false;
                    clickedInfo = null;
                }
            }

            const newData = await updatedPlotRes.json();
            spatialData = JSON.parse(JSON.stringify(newData)); // 强制换引用
            allLog = await updatedLogRes.json();
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

    async function refreshSpatialState() {
        const {
            ncountData,
            plotData,
            sliceInfo,
            metricsData,
            logData,
            cellChatData,
        } = await fetchSpatial();

        spatialData = plotData;
        spatialInfo = sliceInfo;
        currentMethod = sliceInfo.cluster_method;
        epoch = sliceInfo.epoch ?? 500;
        n_clusters = sliceInfo.n_clusters ?? 7;
        ncountSpatialData = ncountData;
        spotMetricsData = metricsData;
        allLog = logData;
        cellChat = JSON.parse(JSON.stringify(cellChatData)); // 或者结构复制

        updateClusterMeta(plotData);
    }

    function updateClusterMeta(plotData) {
        const clusters = new Set();
        plotData.forEach((trace) => clusters.add(trace.name));
        availableClusters = Array.from(clusters);

        const clusterNames = availableClusters.sort(
            (a, b) => +a.replace("Cluster ", "") - +b.replace("Cluster ", ""),
        );

        clusterColorScale = d3
            .scaleOrdinal()
            .domain(clusterNames)
            .range(d3.schemeTableau10);
    }

    async function iniCluster() {
        ncountSpatialData = null;
        spotMetricsData = null;
        allLog = null;
        cellChat = null;
        spatialData = null;
        hvg = {};
        umapData = null;

        const clusterRes = await fetch(baseApi + "/run-clustering", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                n_clusters,
                method: currentMethod,
                slice_id: currentSlice,
                epoch,
            }),
        });

        spatialData = await clusterRes.json();
        umapData = await fecthUmapData();

        await refreshSpatialState();
        console.log(spatialData);
    }

    onMount(async () => {
        Split(["#left-panel", "#main-panel", "#right-panel"], {
            sizes: [17, 50, 33], // 初始宽度比例
            minSize: [150, 300, 200], // 每列的最小宽度
            gutterSize: 4, // 分隔条宽度
            direction: "horizontal",
            cursor: "col-resize",
        });
        await refreshSpatialState();
    });
</script>

{#if !spatialData || !cellChat}
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

<div class="grid h-screen grid-rows-[auto_1fr_auto] gap-y-2 max-h-screen">
    <!-- Header -->
    <header class="text-xl p-3 bg-gray-200">空转数据可视化demo</header>
    <!-- Grid Column -->
    <div
        class="flex grid-cols-1 md:grid-cols-[17%_50%_32%] px-1 gap-x-1 h-full overflow-hidden max-h-full"
        id="split-container"
    >
        <!-- Sidebar (Left) -->
        <aside
            class="p-4 border border-stone-300 rounded-lg text-sm space-y-3 leading-relaxed"
            id="left-panel"
        >
            <!-- Slice 选择器 -->
            <div>
                <label class="block font-semibold text-gray-700 mb-1"
                    >Slice</label
                >
                <select
                    class="w-full border border-gray-300 rounded px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-stone-400"
                    bind:value={currentSlice}
                >
                    {#each allSlices as slice}
                        <option value={slice}>{slice}</option>
                    {/each}
                </select>
            </div>

            <!-- 聚类方法选择器 -->
            <div>
                <label class="block font-semibold text-gray-700 mb-1"
                    >Clustering Method</label
                >
                <select
                    class="w-full border border-gray-300 rounded px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-stone-400"
                    bind:value={currentMethod}
                >
                    {#each clusteringMethods as method}
                        <option value={method}>{method}</option>
                    {/each}
                </select>
            </div>

            <!-- cluster 数量滑块 -->
            <div>
                <label class="block font-semibold text-gray-700 mb-1"
                    >n_clusters</label
                >
                <div class="flex items-center space-x-3">
                    <input
                        type="range"
                        class="flex-1"
                        bind:value={n_clusters}
                        min="3"
                        max="10"
                        step="1"
                    />
                    <span class="w-10 text-right text-sm text-gray-600"
                        >{n_clusters}</span
                    >
                </div>
            </div>

            <!-- Epoch 滑块 -->
            <div>
                <label class="block font-semibold text-gray-700 mb-1"
                    >Epoch</label
                >
                <div class="flex items-center space-x-3">
                    <input
                        type="range"
                        class="flex-1"
                        bind:value={epoch}
                        min="100"
                        max="700"
                        step="50"
                    />
                    <span class="w-10 text-right text-sm text-gray-600"
                        >{epoch}</span
                    >
                </div>
            </div>

            <!-- 聚类按钮 -->
            <div class="pt-2">
                <button
                    type="button"
                    class="btn preset-filled w-full"
                    on:click={iniCluster}
                >
                    Cluster
                </button>
            </div>

            <!-- Spatial Info -->
            {#if spatialInfo}
                <div
                    class="pt-2 border-t border-dashed border-stone-300 text-gray-600"
                >
                    {#each Object.entries(spatialInfo.info_details) as [key, value]}
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
            id="main-panel"
        >
            <Plot
                {spatialData}
                {imageUrl}
                {clusterColorScale}
                {lassoSelected}
                {hoveredBarcode}
                {lassoHover}
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
            class="p-4 border-1 border-solid rounded-lg border-stone-300 h-full overflow-y-scroll scrollbar-none max-h-full"
            style="font-family: sans-serif;scrollbar-width: auto; scrollbar-color: #999 transparent;"
            id="right-panel"
        >
            <!-- <header class="text-xl">Inspection View</header> -->
            <div class="max-h-full h-full">
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
                            on:lassoHover={(e) => {
                                lassoHover = e.detail;
                                console.log("main", e.detail);
                            }}
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
                        class="max-h-full h-full"
                        {spotMetricsData}
                        {clusterColorScale}
                        {allLog}
                        {currentSlice}
                        {baseApi}
                        {hoveredBarcode}
                        {hvg}
                        {availableClusters}
                        {umapData}
                        {cellChat}
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

<script>
    import { createEventDispatcher, onMount } from "svelte";
    import Plotly from "plotly.js-dist-min";
    import { Edit } from "@lucide/svelte";
    import { Tabs } from "@skeletonlabs/skeleton-svelte";
    import * as d3 from "d3";

    export let clickedInfo;
    export let availableClusters;
    export let baseApi;
    export let currentSlice;

    const dispatch = createEventDispatcher();

    let group = "BasicInfo";
    let clusterEdit = false;
    let comment = ""; // 备注
    let selectedCluster = null; // 新选的 cluster
    let log;
    let expandedIndex = null;
    let expression = null;
    let expressionBarDiv = null;

    $: if (clickedInfo) {
        // group = "BasicInfo";
        clusterEdit = false;
        selectedCluster = clickedInfo.cluster;
    }

    async function fetchLog(barcode) {
        const res = await fetch(
            `${baseApi}/cluster-log-by-spot?slice_id=${currentSlice}&barcode=${barcode}`,
        );
        log = await res.json();
    }

    function drawExpressionBar() {
        const genes = clickedInfo.expression.map(([g]) => g);
        const values = clickedInfo.expression.map(([, v]) => v);
    }

    async function fetchExpression(barcode) {
        const res = await fetch(`${baseApi}/expression?barcode=${barcode}`);
        const expression = await res.json();
        clickedInfo.expression = Object.entries(expression)
            .filter(([, v]) => v > 0)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 50);

        const genes = clickedInfo.expression.map(([g]) => g);
        const values = clickedInfo.expression.map(([, v]) => v);
        const data = [
            {
                type: "bar",
                x: values.reverse(),
                y: genes.reverse(),
                orientation: "h",
                marker: {
                    color: "rgba(58, 71, 80, 0.6)",
                    line: {
                        color: "rgba(58, 71, 80, 1.0)",
                        width: 1,
                    },
                },
            },
        ];

        const layout = {
            title: "Gene Expression",
            height: 600,
            margin: { l: 60, r: 30, t: 30, b: 30 },
        };

        Plotly.newPlot(expressionBarDiv, data, layout, {
            responsive: true,
            displayModeBar: false,
            useResizeHandler: true,
            scrollZoom: true,
        });
    }

    $: if (clickedInfo && group === "ChangeLog") {
        fetchLog(clickedInfo.barcode);
    }

    $: if (
        clickedInfo &&
        group === "GeneExpression" &&
        !clickedInfo.expression
    ) {
        fetchExpression(clickedInfo.barcode);
    }

    function changeCluster(barcode, value, comment) {
        // console.log(currentSlice, barcode, clickedInfo.cluster, value, comment);
        clusterEdit = false;
        dispatch("clusterUpdate", {
            barcode,
            newCluster: value,
            oldCluster: clickedInfo.cluster,
            comment,
        });
    }
</script>

<Tabs
    bind:value={group}
    onValueChange={(e) => (group = e.value)}
    class="w-full overflow-x-auto h-full"
>
    {#snippet list()}
        <Tabs.Control value="BasicInfo">Basic</Tabs.Control>
        <Tabs.Control value="GeneExpression">Gene</Tabs.Control>
        <Tabs.Control value="ChangeLog">Log</Tabs.Control>
    {/snippet}
    {#snippet content()}
        <Tabs.Panel value="BasicInfo">
            {#each Object.entries(clickedInfo) as [key, spotValue]}
                {#if key !== "expression"}
                    <!-- 跳过表达数据，单独处理 -->
                    <div class="mb-2">
                        <strong>{key}:</strong>
                        {#if key === "cluster"}
                            {#if !clusterEdit}
                                <div
                                    class="flex flex-row items-center gap-2 mt-1"
                                >
                                    <span>{spotValue}</span>
                                    <button
                                        on:click={() => {
                                            clusterEdit = true;
                                        }}
                                        class="btn btn-sm"
                                    >
                                        <Edit size="15" />
                                    </button>
                                </div>
                            {:else}
                                <div class="mt-2 space-y-3">
                                    <div class="flex items-center gap-2">
                                        <label class="w-24 font-medium text-sm"
                                            >New Cluster:</label
                                        >
                                        <select
                                            class="select flex-1"
                                            bind:value={selectedCluster}
                                        >
                                            {#each availableClusters as cluster}
                                                <option value={cluster}
                                                    >{cluster}</option
                                                >
                                            {/each}
                                        </select>
                                    </div>

                                    <div>
                                        <label
                                            class="block mb-1 text-sm font-medium"
                                            >Comment</label
                                        >
                                        <textarea
                                            class="textarea w-full"
                                            rows="3"
                                            placeholder="Comment"
                                            bind:value={comment}
                                        ></textarea>
                                    </div>

                                    <div class="flex gap-2">
                                        <button
                                            type="button"
                                            class="btn preset-filled"
                                            on:click={() => {
                                                if (
                                                    clickedInfo?.barcode &&
                                                    selectedCluster
                                                ) {
                                                    changeCluster(
                                                        clickedInfo.barcode,
                                                        selectedCluster,
                                                        comment,
                                                    );
                                                    clusterEdit = false;
                                                    comment = "";
                                                }
                                            }}
                                        >
                                            Confirm
                                        </button>
                                        <button
                                            type="button"
                                            class="btn"
                                            on:click={() => {
                                                clusterEdit = false;
                                            }}
                                        >
                                            Cancel
                                        </button>
                                    </div>
                                </div>
                            {/if}
                        {:else}
                            <div class="mt-1">
                                <span>{spotValue}</span>
                            </div>
                        {/if}
                    </div>
                {/if}
            {/each}
        </Tabs.Panel>
        <Tabs.Panel value="GeneExpression" class="h-full flex-grow">
            <div bind:this={expressionBarDiv} class="w-full h-full"></div>
            <!-- {#if clickedInfo && clickedInfo.expression}
                <ul class="list-disc list-inside">
                    {#each Object.entries(clickedInfo.expression)
                        .filter(([_, v]) => v !== 0)
                        .sort((a, b) => b[1] - a[1]) as [gene, value]}
                        <li>
                            {gene}: {value.toFixed(2)}
                        </li>
                    {/each}
                </ul>
            {/if} -->
        </Tabs.Panel>
        <Tabs.Panel value="ChangeLog">
            {#if log}
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
                            {#each log as row, i}
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
                                <td class="text-right">{log.length}</td>
                            </tr>
                        </tfoot>
                    </table>
                </div>
            {/if}
        </Tabs.Panel>
    {/snippet}
</Tabs>

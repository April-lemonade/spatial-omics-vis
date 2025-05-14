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

    $: if (clickedInfo) {
        group = "BasicInfo";
        clusterEdit = false;
        selectedCluster = clickedInfo.cluster;
    }

    async function fetchLog(barcode) {
        const res = await fetch(
            `${baseApi}/cluster-log-by-spot/?slice_id=${currentSlice}&barcode=${barcode}`,
        );
        log = await res.json();
    }

    $: if (clickedInfo && group === "ChangeLog") {
        fetchLog(clickedInfo.barcode);
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
    class="w-full overflow-x-auto"
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
                        <div class="flex flex-row items-center gap-2">
                            {#if key === "cluster"}
                                {#if !clusterEdit}
                                    <span>{spotValue}</span>
                                    <button
                                        on:click={() => {
                                            clusterEdit = true;
                                        }}
                                    >
                                        <Edit size="15" />
                                    </button>
                                {:else}
                                    <form
                                        class="mx-auto w-full max-w-md space-y-4"
                                    >
                                        <select
                                            class="select"
                                            bind:value={selectedCluster}
                                        >
                                            {#each availableClusters as cluster}
                                                <option value={cluster}
                                                    >{cluster}</option
                                                >
                                            {/each}
                                        </select>

                                        <label class="label">
                                            <!-- <span
                                                                class="label-text"
                                                                >Comment</span
                                                            > -->
                                            <textarea
                                                class="textarea"
                                                rows="4"
                                                placeholder="Comment"
                                                bind:value={comment}
                                            ></textarea>
                                        </label>
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
                                            }}>Confirm</button
                                        >
                                        <button
                                            type="button"
                                            class="btn preset-filled"
                                            on:click={() => {
                                                clusterEdit = false;
                                            }}>Cancel</button
                                        >
                                    </form>
                                {/if}
                            {:else}
                                <span>{spotValue}</span>
                            {/if}
                        </div>
                    </div>
                {/if}
            {/each}
        </Tabs.Panel>
        <Tabs.Panel value="GeneExpression">
            {#if clickedInfo && clickedInfo.expression}
                <ul class="list-disc list-inside">
                    {#each Object.entries(clickedInfo.expression)
                        .filter(([_, v]) => v !== 0)
                        .sort((a, b) => b[1] - a[1]) as [gene, value]}
                        <li>
                            {gene}: {value.toFixed(2)}
                        </li>
                    {/each}
                </ul>
            {/if}
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

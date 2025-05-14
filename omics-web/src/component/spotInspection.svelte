<script>
    import { createEventDispatcher, onMount } from "svelte";
    import Plotly from "plotly.js-dist-min";
    import { Edit } from "@lucide/svelte";
    import { Tabs } from "@skeletonlabs/skeleton-svelte";
    import * as d3 from "d3";

    export let clickedInfo;
    export let availableClusters;

    const dispatch = createEventDispatcher();

    let group = "BasicInfo";
    let clusterEdit = false;
    let comment = ""; // 备注
    let selectedCluster = null; // 新选的 cluster

    $: if (clickedInfo) {
        clusterEdit = false;
        selectedCluster = clickedInfo.cluster;
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

<Tabs bind:value={group} onValueChange={(e) => (group = e.value)}>
    {#snippet list()}
        <Tabs.Control value="BasicInfo">Basic Info</Tabs.Control>
        <Tabs.Control value="GeneExpression">Gene Expression</Tabs.Control>
        <Tabs.Control value="ChangeLog">Change Log</Tabs.Control>
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
        <Tabs.Panel value="ChangeLog"></Tabs.Panel>
    {/snippet}
</Tabs>

<script>
    import { ProgressRing } from "@skeletonlabs/skeleton-svelte";
    import { createEventDispatcher } from "svelte";

    export let currentSlice;
    export let baseApi;
    export let clickedInfo;

    const dispatch = createEventDispatcher();
    const methods = ["RF"];
    let currentMethod = methods[0];

    let reclustered = false;
    let reclusering = false;
    let expandedIndex = null;

    $: if (clickedInfo && reclustered) {
        const hasOriginal = clickedInfo?.[0]?.original_cluster !== undefined;
        if (!hasOriginal) {
            reclustered = false;
            reclusering = false;
            expandedIndex = null;
        }
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

    function acceptRecluster(row) {
        console.log(row);
        dispatch("acceptRecluster", {
            barcode: row.barcode,
            newCluster: row.new_cluster,
            oldCluster: row.original_cluster,
            comment: `${currentMethod} Recluster`,
        });
    }
</script>

<div class="h-full">
    {#if reclustered && !reclusering && clickedInfo}
        <!-- <nav
            class="btn-group preset-outlined-surface-200-800 flex-col p-2 md:flex-row"
        >
            <button type="button" class="btn preset-filled">All</button>
            <button type="button" class="btn hover:preset-tonal">Changed</button
            >
        </nav> -->
        <div class="table-wrap">
            <table class="table caption-bottom text-xs w-full h-full">
                <thead>
                    <tr>
                        <th>Barcode</th>
                        <th>Prev</th>
                        <th>New</th>
                        <th>&nbsp;</th>
                    </tr>
                </thead>
                <tbody class="[&>tr]:hover:preset-tonal-primary">
                    {#each clickedInfo as row, i}
                        <tr
                            class="cursor-pointer {row.original_cluster !==
                            row.new_cluster
                                ? 'bg-red-100'
                                : ''}"
                            on:click={() =>
                                (expandedIndex =
                                    expandedIndex === i ? null : i)}
                        >
                            <td>{row.barcode}</td>
                            <td>{row.original_cluster}</td>
                            <td>{row.new_cluster}</td>
                            <td class="text-right">
                                {#if row.original_cluster !== row.new_cluster}
                                    <button
                                        class="btn btn-sm preset-filled"
                                        on:click={(e) => {
                                            e.stopPropagation();
                                            acceptRecluster(clickedInfo[i]);
                                        }}
                                    >
                                        &check;
                                    </button>
                                {/if}
                            </td>
                        </tr>
                        {#if expandedIndex === i}
                            <tr class="bg-muted/30 text-sm">
                                <td colspan="3">
                                    {#each Object.entries(row) as [key, value], i}
                                        {#if i >= 4}
                                            <div>{key}:{value}</div>
                                        {/if}
                                    {/each}
                                </td>
                            </tr>
                        {/if}
                    {/each}
                </tbody>
                <tfoot>
                    <tr>
                        <td colspan="3">Total</td>
                        <td class="text-right">{clickedInfo.length}</td>
                    </tr>
                </tfoot>
            </table>
        </div>
    {:else if clickedInfo && !reclustered}
        <div class="flex flex-col gap-5">
            <!-- 显示选中数量 -->
            <div>{clickedInfo?.length || 0} spots selected</div>

            <!-- 聚类方法选择 -->
            <div>
                <!-- svelte-ignore a11y_label_has_associated_control -->
                <label class="font-bold block mb-1">Clustering method:</label>
                <select class="select w-full" bind:value={currentMethod}>
                    {#each methods as method}
                        <option value={method}>{method}</option>
                    {/each}
                </select>
            </div>

            <!-- 重新聚类按钮 -->
            <div>
                <button
                    type="button"
                    class="btn preset-filled w-full"
                    on:click={recluster}
                    disabled={!clickedInfo?.length}
                >
                    Recluster
                </button>
            </div>
        </div>
    {/if}

    {#if reclusering}
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
</div>

<script>
    import { ProgressRing } from "@skeletonlabs/skeleton-svelte";
    import { createEventDispatcher, tick } from "svelte";
    import Plotly from "plotly.js-dist-min";
    import { Switch } from "@skeletonlabs/skeleton-svelte";

    export let currentSlice;
    export let baseApi;
    export let clickedInfo;

    const dispatch = createEventDispatcher();
    const methods = ["RF"];
    let currentMethod = methods[0];
    let currentRow = null;
    let result;
    let remainingChangedCount;

    let reclustered = false;
    let reclusering = false;
    let expandedIndex = null;
    let lasssoHover = null;
    let state = false;
    let filteredResults;

    $: if (clickedInfo && reclustered) {
        const hasOriginal = clickedInfo?.[0]?.original_cluster !== undefined;
        if (!hasOriginal && remainingChangedCount === 0) {
            reclustered = false;
            reclusering = false;
            expandedIndex = null;
        }
    }

    // $: if (result) {
    //     filteredResults = state
    //         ? result.filter((item) => item.changed === true)
    //         : result;
    //     // if (state) {
    //     //     const filtered = result.filter(item => item.changed === true);
    //     // } else
    // }

    $: filteredResults =
        result && state ? result.filter((item) => item.changed) : result;

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
            result = data;
            remainingChangedCount = result.filter(
                (r) => r.changed === true,
            ).length;
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

        const index = result.findIndex((r) => r.barcode === row.barcode);
        if (index !== -1) {
            result[index].changed = false;
            // 手动触发更新
            result = [...result];
        }

        // 重新计算仍然 changed === true 的个数
        remainingChangedCount = result.filter((r) => r.changed === true).length;

        dispatch("acceptRecluster", {
            barcode: row.barcode,
            newCluster: row.new_cluster,
            oldCluster: row.original_cluster,
            comment: `${currentMethod} Recluster`,
            remainingChangedCount,
        });
    }

    async function drawRadar(row) {
        await tick();

        const categories = [
            "prob_1.0",
            "prob_2.0",
            "prob_3.0",
            "prob_4.0",
            "prob_5.0",
            "prob_6.0",
            "prob_7.0",
        ];
        const values = categories.map((k) => currentRow[k] ?? 0);

        const maxValue = Math.max(...values);
        const paddedMax = Math.min(1, Math.ceil((maxValue + 0.05) * 10) / 10);

        const trace = {
            type: "scatterpolar",
            r: values,
            theta: categories,
            fill: "toself",
            name: currentRow.barcode,
        };

        const layout = {
            dragmode: false,
            margin: { t: 20, b: 20, l: 20, r: 20 },
            polar: {
                radialaxis: { visible: true, range: [0, paddedMax] },
            },
            showlegend: false,
        };

        Plotly.newPlot("radar-" + currentRow.barcode, [trace], layout, {
            scrollZoom: true,
            displaylogo: false,
            displayModeBar: false,
            responsive: true,
        });
    }

    $: if (currentRow) {
        console.log(currentRow);
        drawRadar(currentRow);
    }
</script>

<div class="h-full">
    {#if reclustered && !reclusering && clickedInfo}
        <div class="flex justify-end items-center space-x-2 w-full">
            <span class="text-sm text-gray-700">Changed Only</span>
            <Switch
                name="example"
                checked={state}
                onCheckedChange={(e) => (state = e.checked)}
            />
        </div>

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
                    {#each filteredResults as row, i}
                        <tr
                            class="cursor-pointer {row.changed
                                ? 'bg-red-100'
                                : ''}"
                            on:click={() => {
                                expandedIndex = expandedIndex === i ? null : i;
                                currentRow = row;
                            }}
                            on:mouseenter={() => {
                                console.log("table", row);
                                dispatch("lassoHover", {
                                    barcode: row.barcode,
                                    newCluster: row.new_cluster,
                                });
                            }}
                            on:mouseleave={() => {
                                dispatch("lassoHover", {
                                    barcode: "",
                                    newCluster: "",
                                });
                            }}
                        >
                            <td>{row.barcode}</td>
                            <td>{row.original_cluster}</td>
                            <td>{row.new_cluster}</td>
                            <td class="text-right">
                                {#if row.changed}
                                    <button
                                        class="btn btn-sm preset-filled"
                                        on:click={(e) => {
                                            e.stopPropagation();
                                            acceptRecluster(filteredResults[i]);
                                        }}
                                    >
                                        &check;
                                    </button>
                                {/if}
                            </td>
                        </tr>
                        {#if expandedIndex === i}
                            <tr class="bg-muted/30 text-sm">
                                <td colspan="4">
                                    {#each Object.entries(row) as [key, value], i}
                                        {#if ["p_value", "p_value_refined", "prob_diff"].includes(key)}
                                            <div>{key}:{value}</div>
                                        {/if}
                                    {/each}
                                    <div
                                        id={"radar-" + row.barcode}
                                        class="w-full h-64"
                                    ></div>
                                </td>
                            </tr>
                        {/if}
                    {/each}
                </tbody>
                <tfoot>
                    <tr>
                        <td colspan="3">Total</td>
                        <td class="text-right">{filteredResults.length}</td>
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

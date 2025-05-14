from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import scanpy as sc
import numpy as np

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

dir = ["./VT3D/E16-18h/E16-18h_a_count_normal_stereoseq.h5ad","./VT3D/hypo_preoptic/hypo_preoptic.h5ad"]

# 加载数据
# adata = sc.read_h5ad("./VT3D/E16-18h/E16-18h_a_count_normal_stereoseq.h5ad")

@app.get("/celltype-spatial/{index}")
def get_celltype_spatial(index: int):
    adata = sc.read_h5ad(dir[index])
    labels = adata.obs["annotation"].astype(str).tolist()
    # 提取位置和 cell type 注释
    if index == 0:
        coords = adata.obs[["new_x", "new_y", "new_z"]].values.tolist()      
    else:
        coords = adata.obsm["spatial3D"][:, :3].tolist()
   

    # 为每个 cell type 分配一个 RGB 颜色
    # unique_labels = sorted(set(labels))
    # label_color_map = {
    #     label: [np.random.rand(), np.random.rand(), np.random.rand()]
    #     for label in unique_labels
    # }

    # 构造点和颜色数据
    points = []
    for coord, label in zip(coords, labels):
        points.append({
            "position": coord,
            "label": label
        })

    return {
        "points": points,
    }
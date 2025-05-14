import os

os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import squidpy as sq
import scanpy as sc
import pandas as pd
import json
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import Table, Column, Integer, String, MetaData, TIMESTAMP, Float, func, create_engine
from sqlalchemy.exc import ProgrammingError, IntegrityError
from sqlalchemy import insert, text
from rpy2.robjects import pandas2ri
import re

# 建立连接
engine = create_engine("mysql+pymysql://root:@localhost/omics_data", echo=True)
metadata = MetaData()

pandas2ri.activate()


def insert_initial_clusters(adata, engine, slice_id):
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table_name = f"spot_cluster_{slice_id}"
    spot_cluster = metadata.tables[table_name]

    with engine.connect() as conn:
        for i, (barcode, row) in enumerate(adata.obs.iterrows()):
            n_count = float(row["nCount_Spatial"]) if "nCount_Spatial" in row else None
            cluster = str(row["cluster"])
            n_feature = float(row.get("nFeature_Spatial", None))
            percent_mito = float(row.get("pct_counts_mt", None))
            percent_ribo = float(row.get("pct_counts_ribo", None))
            x, y = map(float, adata.obsm["spatial"][i])
            try:
                conn.execute(
                    insert(spot_cluster).values(
                        barcode=barcode,
                        cluster=cluster,
                        x=x,
                        y=y,
                        n_count_spatial=n_count,
                        n_feature_spatial=n_feature,
                        percent_mito=percent_mito,
                        percent_ribo=percent_ribo,
                    )
                )
            except IntegrityError:
                continue
        conn.commit()


def create_tables(slice_id):
    table_name = f"spot_cluster_{slice_id}"
    spot_cluster = Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("barcode", String(50), nullable=False, unique=True),
        Column("cluster", String(50), nullable=False),
        Column("x", Float),
        Column("y", Float),
        Column("n_count_spatial", Float),
        Column("n_feature_spatial", Float),
        Column("percent_mito", Float),
        Column("percent_ribo", Float),
        Column("updated_at", TIMESTAMP, server_default=func.now(), onupdate=func.now()),
    )

    cluster_log = Table(
        "cluster_log",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("slice_id", String(50), nullable=False),
        Column("barcode", String(50), nullable=False),
        Column("old_cluster", String(50), nullable=False),
        Column("new_cluster", String(50), nullable=False),
        Column("comment", String(255)),
        Column("updated_at", TIMESTAMP, server_default=func.now(), onupdate=func.now()),
    )

    try:
        metadata.create_all(engine)
    except ProgrammingError as e:
        print(f"Error creating tables: {e}")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="./data/151673/spatial"), name="images")

# 全局路径与缓存
slice_id = "151673"
path = f"./data/{slice_id}"
spatial_dir = os.path.join(path, "spatial")
# spatial_zip = os.path.join(path, "VISDS000003_spatial.zip")
# deco_csv = os.path.join(path, "VISDS000003_celltype_deco.csv")
# DEG_rds = os.path.join(path, "VISDS000003_all_cluster_DEG.rds")

adata = None
expression_data = None


def prepare_data():
    global adata
    if adata is not None:
        return adata

    create_tables(slice_id)

    adata_local = sq.read.visium(path=path)
    adata_local.obs["nCount_Spatial"] = (
        adata_local.X.sum(axis=1).A1 if hasattr(adata_local.X, "A1") else adata_local.X.sum(axis=1)
    )
    adata_local.obs["nFeature_Spatial"] = (adata_local.X > 0).sum(1).A1 if hasattr(adata_local.X, "A1") else (
                adata_local.X > 0).sum(1)

    adata_local.var["mt"] = adata_local.var_names.str.startswith("MT-")
    adata_local.var["ribo"] = adata_local.var_names.str.startswith(("RPS", "RPL"))
    sc.pp.calculate_qc_metrics(adata_local, qc_vars=["mt", "ribo"], inplace=True)

    gene_names = adata_local.var_names.tolist()
    X = adata_local.X.toarray() if hasattr(adata_local.X, "toarray") else adata_local.X
    global expression_data
    expression_data = [dict(zip(gene_names, map(float, X[i]))) for i in range(X.shape[0])]

    sc.pp.filter_genes(adata_local, min_cells=3)
    sc.pp.normalize_total(adata_local)
    sc.pp.log1p(adata_local)
    sc.pp.highly_variable_genes(adata_local, flavor="seurat", n_top_genes=2000)
    adata_local = adata_local[:, adata_local.var.highly_variable]
    sc.pp.pca(adata_local)
    sc.pp.neighbors(adata_local)
    sc.tl.leiden(adata_local, key_added="cluster")

    adata = adata_local
    insert_initial_clusters(adata, engine, slice_id)
    return adata


@app.get("/allslices")
def get_all_slice_ids(data_root="./data"):
    return [
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name)) and name.isdigit()
    ]


@app.on_event("startup")
def load_once():
    prepare_data()


@app.get("/plot-data")
def get_plot_data(slice_id: str = Query(...), scale: str = "hires"):
    # 加载 scalefactor
    spatial_dir = os.path.join(f"./data/{slice_id}", "spatial")
    with open(os.path.join(spatial_dir, "scalefactors_json.json"), "r") as f:
        sf = json.load(f)
    scale_key = "tissue_hires_scalef" if scale == "hires" else "tissue_lowres_scalef"
    factor = sf[scale_key]

    # 从数据库读取 cluster 和坐标信息
    table_name = f"spot_cluster_{slice_id}"
    query = text(f"SELECT barcode, cluster, x, y FROM `{table_name}`")

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    # 构造 DataFrame
    df = pd.DataFrame(rows, columns=["barcode", "cluster", "x", "y"])

    # 应用缩放因子
    df["x"] = df["x"] * factor
    df["y"] = df["y"] * factor

    # 构造 Plotly traces
    traces = []
    for cluster_id, group in df.groupby("cluster"):
        trace = {
            "x": group["x"].tolist(),
            "y": group["y"].tolist(),
            "name": f"Cluster {cluster_id}",
            "type": "scatter",
            "mode": "markers",
            "customdata": group["barcode"].tolist(),
            "hovertemplate": "Barcode: %{customdata}<extra></extra>",
        }
        traces.append(trace)

    return traces


@app.get("/expression/{barcode}")
def get_expression(barcode: str):
    global adata
    if barcode not in adata.obs_names:
        raise HTTPException(status_code=404, detail="Barcode not found")
    i = adata.obs_names.get_loc(barcode)
    gene_names = adata.var_names.tolist()
    expr = adata.X[i].toarray().flatten() if hasattr(adata.X, "toarray") else adata.X[i]
    return dict(zip(gene_names, map(float, expr)))


@app.get("/slice-info")
def get_slice_info(slice_id: str = Query(..., description="Slide ID like 151673")):
    path = f"./data/{slice_id}"
    info_path = os.path.join(path, "info.json")

    if not os.path.exists(info_path):
        raise HTTPException(status_code=404, detail="info.json not found for this slice")

    # 读取基础信息
    with open(info_path, "r") as f:
        info = json.load(f)

    # 实时加载 adata 以获取统计量
    adata_local = sq.read.visium(path=path)
    sc.pp.filter_genes(adata_local, min_cells=3)
    sc.pp.normalize_total(adata_local)
    sc.pp.log1p(adata_local)

    info.update({
        "spot_count": adata_local.n_obs,
        "gene_count": adata_local.n_vars,
        "avg_genes_per_spot": round(float((adata_local.X > 0).sum(1).mean()), 2)
    })

    return info



@app.get("/ncount_by_cluster")
def get_ncount_by_cluster(slice_id: str = Query(...)):
    table_name = f"spot_cluster_{slice_id}"
    query = text(f"SELECT cluster, n_count_spatial FROM `{table_name}`")

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    # 聚类分组
    cluster_dict = {}
    for cluster, ncount in rows:
        cluster_name = f"Cluster {cluster}"
        cluster_dict.setdefault(cluster_name, []).append(ncount)

    return cluster_dict


@app.get("/spot-metrics")
def get_spot_metrics(slice_id: str = Query(...)):
    table_name = f"spot_cluster_{slice_id}"
    query = text(f"""
        SELECT barcode, cluster, 
               n_count_spatial AS nCount_Spatial,
               n_feature_spatial AS nFeature_Spatial,
               percent_mito,
               percent_ribo
        FROM `{table_name}`
        WHERE n_count_spatial IS NOT NULL
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    df = pd.DataFrame(rows, columns=[
        "barcode", "cluster", "nCount_Spatial", "nFeature_Spatial", "percent_mito", "percent_ribo"
    ])

    # 转换为长格式，每个指标一行，便于前端分面绘图
    long_df = df.melt(
        id_vars=["barcode", "cluster"],
        value_vars=["nCount_Spatial", "nFeature_Spatial", "percent_mito", "percent_ribo"],
        var_name="metric",
        value_name="value"
    )

    return long_df.to_dict(orient="records")


class ClusterUpdateRequest(BaseModel):
    slice_id: str
    barcode: str
    old_cluster: str
    new_cluster: str
    comment: str = ""


@app.post("/update-cluster")
def update_cluster(req: ClusterUpdateRequest):
    table_name = f"spot_cluster_{req.slice_id}"

    with engine.begin() as conn:
        # 校验 barcode 是否存在且原始 cluster 一致
        result = conn.execute(
            text(f"SELECT cluster FROM `{table_name}` WHERE barcode = :barcode"),
            {"barcode": req.barcode}
        ).fetchone()

        # clusterNumbers = oldClusterStrings.map(s => parseInt(s.replace(/\D/g, '')));

        if result is None:
            raise HTTPException(status_code=404, detail="Barcode not found")

        if result[0] != re.search(r'\d+', req.old_cluster).group():
            raise HTTPException(status_code=400, detail="Old cluster does not match current value")

        # 更新 cluster
        conn.execute(
            text(f"""
                UPDATE `{table_name}`
                SET cluster = :new_cluster
                WHERE barcode = :barcode
            """),
            {"new_cluster": re.search(r'\d+', req.new_cluster).group(), "barcode": req.barcode}
        )

        # 写入日志表
        conn.execute(
            text(f"""
                INSERT INTO cluster_log (slice_id, barcode, old_cluster, new_cluster, comment)
                VALUES (:slice_id, :barcode, :old_cluster, :new_cluster, :comment)
            """),
            req.dict()
        )

    return {"message": "Cluster updated and logged successfully."}


@app.get("/cluster-log")
def get_cluster_log(slice_id: str = Query(...)):
    query = text("""
        SELECT barcode, old_cluster, new_cluster, comment, updated_at
        FROM cluster_log
        WHERE slice_id = :slice_id
        ORDER BY updated_at DESC
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, {"slice_id": slice_id}).fetchall()

    return [
        {
            "barcode": r[0],
            "old_cluster": r[1],
            "new_cluster": r[2],
            "comment": r[3],
            "updated_at": r[4].isoformat() if r[4] else None
        }
        for r in rows
    ]

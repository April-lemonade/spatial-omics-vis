import os

import torch

os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import squidpy as sq
import scanpy as sc
import pandas as pd
import json
from sqlalchemy import Table, Column, Integer, String, MetaData, TIMESTAMP, Float, func, create_engine, insert, text,Text,select
from sqlalchemy.exc import ProgrammingError
from rpy2.robjects import pandas2ri
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import gseapy as gp
from typing import List
from dotenv import load_dotenv
from GraphST.utils import clustering
from GraphST import GraphST
from sqlalchemy.dialects.mysql import insert as mysql_insert
from scipy.spatial import Delaunay


load_dotenv()

user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
db = os.getenv("DB_NAME")



# 建立连接
engine = create_engine("mysql+pymysql://root:@localhost/omics_data", echo=True)
# engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db}", echo=True)

# engine = create_engine(
#     f"mysql+pymysql://{user}:{password}@{host}/{db}",
#     echo=True,
#     pool_recycle=3600,  # 防止 MySQL 超时自动断开连接
#     pool_pre_ping=True,  # 自动检查连接是否有效
#     connect_args={"connect_timeout": 30}  # 设置连接超时为 30 秒
# )
metadata = MetaData()

pandas2ri.activate()


def insert_initial_clusters(adata, engine, slice_id):
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table_name = f"spot_cluster_{slice_id}"
    spot_cluster = metadata.tables[table_name]

    with engine.connect() as conn:
        result = conn.execute(spot_cluster.select().limit(1)).fetchone()
        if result is not None:
            print(f"Table {table_name} already has data. Skipping insertion.")
            return

        # ✅ 构造插入列表
        records = []
        for i, (barcode, row) in enumerate(adata.obs.iterrows()):
            x, y = map(float, adata.obsm["spatial"][i])
            emb_vec = adata.obsm["emb"][i]
            emb_str = ",".join(map(str, emb_vec))  # 将嵌入向量转为字符串

            records.append({
                "barcode": barcode,
                "cluster": str(row["domain"]),
                "x": x,
                "y": y,
                "n_count_spatial": float(row.get("nCount_Spatial", None)),
                "n_feature_spatial": float(row.get("nFeature_Spatial", None)),
                "percent_mito": float(row.get("pct_counts_mt", None)),
                "percent_ribo": float(row.get("pct_counts_ribo", None)),
                "emb": emb_str,
            })

        # ✅ 批量插入
        conn.execute(insert(spot_cluster), records)
        conn.commit()
        print(f"✅ 批量插入 {len(records)} 条记录到 {table_name}")
        
        
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
        Column("emb",Text),
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
    
    cluster_method = Table(
        "cluster_method",
        metadata,
        Column("slice_id", String(50), primary_key=True),
        Column("method", String(50), nullable=False),
        Column("n_clusters", Integer),
        Column("epoch", Integer),
        Column("updated_at", TIMESTAMP, server_default=func.now(), onupdate=func.now())
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

# app.mount("/images", StaticFiles(directory="./data/151673/spatial"), name="images")

# 全局路径与缓存
slice_id = ""
path = ""
scale ="hires"
spatial_dir = ""
sf = None
scale_key = "" 
factor = None

adata = None
expression_data = None


@app.get("/images/{slice_id}/tissue_hires_image.png")
def get_image(slice_id: str):
    path = f"./data/{slice_id}/spatial/tissue_hires_image.png"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png")

def prepare_data():
    '''
    初始只加载spot坐标和基础信息
    '''
    global adata,path
    global slice_id
    if adata is not None:
        return adata
    create_tables(slice_id)

    # 1. 加载 Visium 数据
    adata_local = sq.read.visium(path=path)
    adata_local.obs["nCount_Spatial"] = (
        adata_local.X.sum(axis=1).A1 if hasattr(adata_local.X, "A1") else adata_local.X.sum(axis=1)
    )
    adata_local.obs["nFeature_Spatial"] = (
        (adata_local.X > 0).sum(1).A1 if hasattr(adata_local.X, "A1") else (adata_local.X > 0).sum(1)
    )
    adata_local.var["mt"] = adata_local.var_names.str.startswith("MT-")
    adata_local.var["ribo"] = adata_local.var_names.str.startswith(("RPS", "RPL"))
    sc.pp.calculate_qc_metrics(adata_local, qc_vars=["mt", "ribo"], inplace=True)
    sc.pp.normalize_total(adata_local)
    sc.pp.log1p(adata_local)
    sc.pp.highly_variable_genes(adata_local, flavor="seurat", n_top_genes=2000)

    table_name = f"spot_cluster_{slice_id}"

    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`")).scalar()

        if not result or result == 0:
            # 没有数据时插入记录：cluster = unknown
            print(f"⚠️ 数据库为空，插入初始 spot 记录，cluster='unknown'")
            metadata = MetaData()
            metadata.reflect(bind=engine)
            spot_cluster = metadata.tables[table_name]

            records = []
            for i, (barcode, row) in enumerate(adata_local.obs.iterrows()):
                x, y = map(float, adata_local.obsm["spatial"][i])
                records.append({
                    "barcode": barcode,
                    "cluster": "unknown",
                    "x": x,
                    "y": y,
                    "n_count_spatial": float(row.get("nCount_Spatial", None)),
                    "n_feature_spatial": float(row.get("nFeature_Spatial", None)),
                    "percent_mito": float(row.get("pct_counts_mt", None)),
                    "percent_ribo": float(row.get("pct_counts_ribo", None)),
                    "emb": "",  # 空字符串
                })
            conn.execute(insert(spot_cluster), records)
            conn.commit()
            print(f"✅ 插入 {len(records)} 条记录，等待前端触发聚类")
            adata_local.obs["domain"] = "unknown"
            adata = adata_local
            return adata

        # 已有记录：从数据库恢复
        print(f"✅ 数据库已有记录（{result} 条），加载聚类和 embedding 信息")
        df = pd.read_sql(text(f"SELECT * FROM `{table_name}`"), conn).set_index("barcode")

        for col in ["cluster", "x", "y", "n_count_spatial", "n_feature_spatial", "percent_mito", "percent_ribo"]:
            if col in df.columns:
                if col == "cluster":
                    adata_local.obs[col] = df.loc[adata_local.obs_names, col].astype(str)
                else:
                    adata_local.obs[col] = df.loc[adata_local.obs_names, col].astype(float)

        if "emb" in df.columns:
            def safe_parse(s):
                try:
                    vec = np.fromstring(s, sep=",")
                    return vec if len(vec) > 0 else None
                except:
                    return None
            emb_matrix = df["emb"].apply(safe_parse).dropna()
            if len(emb_matrix) > 0:
                adata_local = adata_local[emb_matrix.index]
                adata_local.obsm["emb"] = np.vstack(emb_matrix.values)

        adata_local.obs["domain"] = adata_local.obs["cluster"].astype("category")
        adata = adata_local
        return adata


class ClusteringRequest(BaseModel):
    slice_id: str
    n_clusters: int = 7
    method: str = "mclust"
    epoch: int = 500
    
@app.post("/run-clustering")
def run_clustering(request: ClusteringRequest):
    '''
    根据前端设置的聚类方法和参数进行聚类
    '''
    global adata
    adata_local = adata.copy()
    if adata is None:
        raise HTTPException(status_code=500, detail="adata 未加载")

    # 👇 执行 GraphST 聚类
    adata_local = run_graphst_and_clustering(adata_local, n_clusters=request.n_clusters, method=request.method,epoch=request.epoch)
    
    adata = adata_local

    # ✅ 批量更新数据库
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table_name = f"spot_cluster_{request.slice_id}"
    spot_cluster = metadata.tables[table_name]

    # 👇 构造批量更新数据（列表形式）
    update_data = []
    for i, (barcode, row) in enumerate(adata_local.obs.iterrows()):
        cluster = f"{float(row['domain']):.1f}"
        emb_vec = adata_local.obsm["emb"][i]
        emb_str = ",".join(map(str, emb_vec))
        update_data.append({
            "barcode": barcode,
            "cluster": cluster,
            "emb": emb_str
        })

    # 👇 使用 insert...on_duplicate_key_update 并传入 update_data
    with engine.begin() as conn:
        insert_stmt = mysql_insert(spot_cluster)
        stmt = insert_stmt.on_duplicate_key_update(
            cluster=insert_stmt.inserted.cluster,
            emb=insert_stmt.inserted.emb
        )
        conn.execute(stmt, update_data)  # ✅ 一定要传第二个参数
        

    print(f"✅ 聚类完成，已批量更新 {len(update_data)} 条记录至 {table_name}")
    with engine.begin() as conn:
         # 👇 删除 cluster_log 中当前 slice_id 的所有记录
        cluster_log = Table("cluster_log", metadata, autoload_with=engine)
        delete_stmt = cluster_log.delete().where(cluster_log.c.slice_id == request.slice_id)
        conn.execute(delete_stmt)
        
        cluster_method = Table("cluster_method", metadata, autoload_with=engine)

        insert_stmt = mysql_insert(cluster_method).values({
            "slice_id": request.slice_id,
            "method": request.method,
            "n_clusters": request.n_clusters,
            "epoch": request.epoch,
        })

        stmt = insert_stmt.on_duplicate_key_update(
            method=insert_stmt.inserted.method,
            n_clusters=insert_stmt.inserted.n_clusters,
            epoch=insert_stmt.inserted.epoch,
            updated_at=func.now()
        )

        conn.execute(stmt)
    
    return get_plot_data(request.slice_id)


def run_graphst_and_clustering(adata_local, n_clusters=7, radius=50, method="mclust", refinement=False,epoch = 500):
    print("⚙️ 执行 GraphST 模型训练与聚类...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GraphST.GraphST(adata_local, device=device, epochs=epoch)
    adata_local = model.train()

    if "emb" in adata_local.obsm:
        print("✅ GraphST 输出维度:", adata_local.obsm["emb"].shape)
    else:
        print("❌ 没有发现 obsm['emb']，聚类可能失败")

    clustering(adata_local, n_clusters=n_clusters, radius=radius, method=method, refinement=refinement)

    adata_local.obs["domain"] = adata_local.obs["domain"].astype(float).map(lambda x: f"{x:.1f}")
    adata_local.obs["domain"] = adata_local.obs["domain"].astype("category")
    adata_local.obs["leiden_original"] = adata_local.obs["domain"].copy()

        

    return adata_local

# def prepare_data_old():
#     global adata
#     if adata is not None:
#         return adata

#     create_tables(slice_id)

#     adata_local = sq.read.visium(path=path)
#     adata_local.obs["nCount_Spatial"] = (
#         adata_local.X.sum(axis=1).A1 if hasattr(adata_local.X, "A1") else adata_local.X.sum(axis=1)
#     )
#     adata_local.obs["nFeature_Spatial"] = (
#         (adata_local.X > 0).sum(1).A1 if hasattr(adata_local.X, "A1") else (adata_local.X > 0).sum(1)
#     )
#     adata_local.var["mt"] = adata_local.var_names.str.startswith("MT-")
#     adata_local.var["ribo"] = adata_local.var_names.str.startswith(("RPS", "RPL"))
#     sc.pp.calculate_qc_metrics(adata_local, qc_vars=["mt", "ribo"], inplace=True)
    
#     sc.pp.normalize_total(adata_local)
#     sc.pp.log1p(adata_local)
#     sc.pp.highly_variable_genes(adata_local, flavor="seurat", n_top_genes=2000)

#     table_name = f"spot_cluster_{slice_id}"
#     with engine.connect() as conn:
#         result = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`")).scalar()
#         if result and result > 0:
#             print(f"✅ 数据库中已有聚类数据（{result} 条），从数据库恢复聚类标签。")
#             df = pd.read_sql(text(f"SELECT * FROM `{table_name}`"), conn).set_index("barcode")

#             # 合并数据库中的聚类信息进 adata.obs
#             for col in ["cluster", "x", "y", "n_count_spatial", "n_feature_spatial", "percent_mito", "percent_ribo"]:
#                 if col in df.columns:
#                     # adata_local.obs[col] = df.loc[adata_local.obs_names, col].astype(str if col == "cluster" else float)
#                     if col == "cluster":
#                         # 👇 保留一位小数（即使是 1.0 也不会变成 1）
#                         adata_local.obs[col] = df.loc[adata_local.obs_names, col].apply(lambda x: f"{float(x):.1f}")
#                     else:
#                         adata_local.obs[col] = df.loc[adata_local.obs_names, col].astype(float)

#             # 如果数据库中存了 emb，也一并恢复
#             if "emb" in df.columns:
#                 def safe_parse(s):
#                     try:
#                         vec = np.fromstring(s, sep=",")
#                         if vec.size == 2000:  # 要与你的 GraphST 输出一致
#                             return vec
#                         else:
#                             print(f"❌ 嵌入维度不一致: {vec.size}")
#                             return None
#                     except:
#                         return None

#                 emb_matrix = df["emb"].apply(safe_parse)
#                 emb_matrix = emb_matrix.dropna()  # 去除解析失败的行
#                 adata_local = adata_local[emb_matrix.index]  # 同步过滤 adata_local
#                 adata_local.obsm["emb"] = np.vstack(emb_matrix.values)

#             # 恢复为主用的 domain 字段
#             adata_local.obs["domain"] = adata_local.obs["cluster"].astype("category")

#             adata = adata_local
#             return adata

#     # ⚙️ 否则执行训练
#     print("⚠️ 数据库无聚类记录，执行 GraphST + 聚类 + 入库...")
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = GraphST.GraphST(adata_local, device=device, epochs=600)
#     adata_local = model.train()
#     # 👇 立即检查输出维度
#     if "emb" in adata_local.obsm:
#         print("✅ GraphST 输出维度:", adata_local.obsm["emb"].shape)
#     else:
#         print("❌ 没有发现 obsm['emb']")
    
#     clustering(adata_local, n_clusters=7, radius=50, method="mclust", refinement=False)
    
#     adata_local.obs["leiden_original"] = adata_local.obs["domain"].copy()
#     adata_local.obs["domain"] = adata_local.obs["domain"].astype("category")
#     if "domain" in adata_local.obs and not pd.api.types.is_categorical_dtype(adata_local.obs["domain"]):
#         print("ℹ️ 将 domain 字段转换为 categorical 类型")
#         adata_local.obs["domain"] = adata_local.obs["domain"].astype("category")
    
#     insert_initial_clusters(adata_local, engine, slice_id)
#     adata = adata_local
#     return adata


@app.get("/allslices")
def get_all_slice_ids(data_root="./data"):
    return [
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name)) and name.isdigit()
    ]


@app.on_event("startup")
def load_once():
    global slice_id,spatial_dir,sf,scale_key,factor,path
    slice_id = get_all_slice_ids()[0]
    spatial_dir = os.path.join(f"./data/{slice_id}", "spatial")
    with open(os.path.join(spatial_dir, "scalefactors_json.json"), "r") as f:
        sf = json.load(f)
    scale_key = "tissue_hires_scalef" if scale == "hires" else "tissue_lowres_scalef"
    path = f"./data/{slice_id}"
    factor = sf[scale_key]
    prepare_data()

    
@app.get("/plot-data")
def get_plot_data(slice_id: str = Query(...)):
    global factor
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
            "name": cluster_id,
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
    adata_local = adata.copy()
    if barcode not in adata.obs_names:
        raise HTTPException(status_code=404, detail="Barcode not found")
    i = adata_local.obs_names.get_loc(barcode)
    gene_names = adata_local.var_names.tolist()
    expr = adata_local.X[i].toarray().flatten() if hasattr(adata_local.X, "toarray") else adata_local.X[i]
    return dict(zip(gene_names, map(float, expr)))


@app.get("/slice-info")
def get_slice_info(slice_id: str = Query(..., description="Slide ID like 151673")):
    path = f"./data/{slice_id}"
    info_path = os.path.join(path, "info.json")

    if not os.path.exists(info_path):
        raise HTTPException(status_code=404, detail="info.json not found for this slice")

    # 读取 info.json 中的原始信息
    with open(info_path, "r") as f:
        info = json.load(f)

    # 加载 adata 并计算统计信息
    adata_local = sq.read.visium(path=path)
    sc.pp.filter_genes(adata_local, min_cells=3)
    sc.pp.normalize_total(adata_local)
    sc.pp.log1p(adata_local)

    # 添加基础统计信息
    info.update({
        "spot_count": adata_local.n_obs,
        "gene_count": adata_local.n_vars,
        "avg_genes_per_spot": round(float((adata_local.X > 0).sum(1).mean()), 2)
    })

    # 查询聚类方法表
    metadata = MetaData()
    metadata.reflect(bind=engine)
    cluster_method_table = metadata.tables["cluster_method"]

    with engine.connect() as conn:
        stmt = select(
            cluster_method_table.c.method,
            cluster_method_table.c.n_clusters,
            cluster_method_table.c.epoch
        ).where(cluster_method_table.c.slice_id == slice_id)
        result = conn.execute(stmt).fetchone()

    # 准备聚类方法字段（保留在顶层）
    cluster_method = result.method if result else "not_clustered"
    
    n_clusters = result.n_clusters if result else None
    epoch = result.epoch if result else None

    # 把其余所有 info 字段打包
    return {
        "cluster_method": cluster_method,
        "n_clusters": n_clusters,
        "epoch": epoch,
        "info_details": info
    }

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

        if result[0] != req.old_cluster:
            raise HTTPException(status_code=400, detail="Old cluster does not match current value")

        # 更新 cluster
        conn.execute(
            text(f"""
                UPDATE `{table_name}`
                SET cluster = :new_cluster
                WHERE barcode = :barcode
            """),
            {"new_cluster": req.new_cluster, "barcode": req.barcode}
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


@app.get("/cluster-log-by-spot")
def get_cluster_log(slice_id: str = Query(...),barcode: str = Query(...)):
    query = text("""
        SELECT barcode, old_cluster, new_cluster, comment, updated_at
        FROM cluster_log
        WHERE slice_id = :slice_id AND barcode = :barcode
        ORDER BY updated_at DESC
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, {"slice_id": slice_id, "barcode": barcode}).fetchall()

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

class selectedBarcodes(BaseModel):
    slice_id: str
    barcode: List[str]
    
@app.post("/recluster")
def recluster(req: selectedBarcodes,factor = factor):
    # print(req.barcode)
    global adata
    print(adata)
    all_barcodes = adata.obs.index.tolist()
    selected_barcodes = req.barcode
    selected_barcodes = list(selected_barcodes)  # 确保是列表类型
    print(selected_barcodes)
    try:
        # 1. 从数据库加载当前的聚类标签
        table_name = f"spot_cluster_{req.slice_id}"
        with engine.connect() as conn:
            db_clusters = pd.read_sql(
                text(f"SELECT barcode, cluster FROM `{table_name}`"),
                conn,
            ).set_index("barcode")

        all_barcodes = adata.obs.index.tolist()
        # selected_barcodes = np.random.choice(all_barcodes, size=, replace=False)
        # selected_barcodes = list(selected_barcodes)
        all_features = pd.DataFrame(adata.obsm['emb'], index=adata.obs.index)
        all_features['cluster'] = adata.obs['domain']

        train_mask = ~all_features.index.isin(selected_barcodes)
        predict_mask = all_features.index.isin(selected_barcodes)

        train_data = all_features[train_mask].copy()
        predict_data = all_features[predict_mask].copy()

        predict_data_original = predict_data[['cluster']].copy()

        feature_cols = [col for col in train_data.columns if col != 'cluster']
        X_train = train_data[feature_cols]
        y_train = train_data['cluster']
        X_predict = predict_data[feature_cols]

        param_grid = {
            'n_estimators': [200],
            'max_depth': [None],
            'min_samples_split': [2]
        }
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='balanced_accuracy')
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_

        # 识别最重要的特征
        importances = best_rf.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        new_clusters = best_rf.predict(X_predict)

        # 获取预测的概率值（所有类别的概率分布）
        prediction_probs = best_rf.predict_proba(X_predict)
        # 获取每个预测的置信度 (最高概率值)
        confidence_scores = np.max(prediction_probs, axis=1)

        # 创建概率分布DataFrame，包含所有类别的概率
        class_names = best_rf.classes_
        prob_cols = [f"prob_{cls}" for cls in class_names]
        probs_df = pd.DataFrame(prediction_probs, columns=prob_cols, index=predict_data.index)

        # 创建结果跟踪DataFrame - 确保使用预测数据的索引
        result_index = predict_data.index
        cluster_change_df = pd.DataFrame(index=result_index)
        cluster_change_df['barcode'] = result_index
        cluster_change_df['original_cluster'] = predict_data_original['cluster'].values
        cluster_change_df['new_cluster'] = new_clusters
        cluster_change_df['confidence'] = confidence_scores

        # 将概率分布添加到结果中 - 使用索引合并而不是concat
        for col in probs_df.columns:
            cluster_change_df[col] = probs_df[col]

        # 添加变化状态列
        cluster_change_df['changed'] = cluster_change_df['original_cluster'] != cluster_change_df['new_cluster']

        # 添加p值列 (1 - 置信度可以近似为p值)
        cluster_change_df['p_value'] = 1 - cluster_change_df['confidence']
        cluster_change_df
       
        adata.uns["change_df"] = cluster_change_df
       
        # 确保目录存在
        os.makedirs('visulization', exist_ok=True)
        
        # 保存结果
        try:
            output_path = f"{slice_id}_processed.h5ad"  # 直接保存在当前目录
            adata.write_h5ad(output_path)
            print(f"成功保存到 {output_path}")
        except Exception as e:
            print(f"保存h5ad文件时出错: {str(e)}")
            # 尝试保存为不同格式
            try:
                # 移除可能导致问题的大型数据
                adata_backup = adata.copy()
                # 移除复杂对象
                if 'uns' in adata_backup.__dict__:
                    for key in list(adata_backup.uns.keys()):
                        if 'spatial' in key or 'domain' in key:
                            del adata_backup.uns[key]
                
                # 保存到绝对路径
                output_path = f"{slice_id}_processed_backup.h5ad"
                adata_backup.write_h5ad(output_path)
                print(f"成功保存到 {output_path}")
                
                # 再次尝试保存完整数据为其他格式
                adata.write_loom(f"{slice_id}_complete.loom")
                print(f"完整数据已保存为 {slice_id}_complete.loom")
            except Exception as e2:
                print(f"备用保存方法也失败: {str(e2)}")
                # 最后尝试保存关键信息
                obs_df = adata.obs.copy()
                obs_df.to_csv(f"{slice_id}_obs_data.csv")
                print(f"已将观测数据保存到 {slice_id}_obs_data.csv")
                
        
        
        # obs = adata.obs
        obs = adata.uns['change_df']
        
        obs_cleaned = obs.replace({np.nan: None, np.inf: None, -np.inf: None})

        return obs_cleaned.reset_index().to_dict(orient="records")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        
def extract_features(adata, barcodes=None, use_hvg_only=True, n_pcs=20):
    """
    从adata中提取特征，包括:
    1. 标准化的空间坐标 (spatial)
    2. 高变基因表达主成分 (如果指定)
    3. 或所有基因的PCA
    
    所有特征都会被适当标准化
    """
    # 如果没有指定barcodes，则使用所有条码
    if barcodes is None:
        barcodes = adata.obs.index.tolist()
        
    # 确保所有条码都在adata中
    valid_barcodes = [b for b in barcodes if b in adata.obs.index]
    if len(valid_barcodes) == 0:
        raise ValueError("没有有效的条码!")
    
    feature_dfs = []
    
    # 1. 提取并标准化空间坐标
    spatial_coords = pd.DataFrame(adata.obsm['spatial'][adata.obs.index.isin(valid_barcodes)], 
                                 index=[b for b in valid_barcodes])
    
    # 标准化空间坐标
    scaler_spatial = StandardScaler()
    spatial_scaled = scaler_spatial.fit_transform(spatial_coords)
    spatial_df = pd.DataFrame(
        spatial_scaled,
        columns=['x_scaled', 'y_scaled'],
        index=valid_barcodes
    )
    feature_dfs.append(spatial_df)
    print(f"提取了空间坐标特征 (标准化)")
    
    # 2. 提取基因表达数据
    # 使用已经计算好的PCA结果
    if 'X_pca' in adata.obsm:
        pca_df = pd.DataFrame(
            adata.obsm['X_pca'][adata.obs.index.isin(valid_barcodes)],
            columns=[f'PC{i+1}' for i in range(adata.obsm['X_pca'].shape[1])],
            index=valid_barcodes
        )
        feature_dfs.append(pca_df)
        print(f"提取了 {pca_df.shape[1]} 个PCA主成分")
    
    # 合并所有特征
    if not feature_dfs:
        raise ValueError("没有提取到任何有效特征!")
        
    features = pd.concat(feature_dfs, axis=1)
    print(f"最终特征矩阵形状: {features.shape}")
    return features

@app.get("/umap-coordinates")
def get_umap_coordinates(slice_id: str = Query(...)):
    global adata
    adata_local = adata.copy()

    if "X_umap" not in adata_local.obsm:
        sc.pp.normalize_total(adata_local)
        sc.pp.log1p(adata_local)
        sc.pp.highly_variable_genes(adata_local, flavor="seurat", n_top_genes=2000)
        adata_local = adata_local[:, adata_local.var.highly_variable]
        sc.pp.pca(adata_local)
        # 使用 adata.obsm["emb"] 构建邻接图
        sc.pp.neighbors(adata_local, use_rep="emb")

        # 基于该邻接图计算 UMAP
        sc.tl.umap(adata_local)

    
    df = pd.DataFrame(
        adata_local.obsm["X_umap"],
        index=adata_local.obs_names,  
        columns=["UMAP_1", "UMAP_2"]
    )

    df["barcode"] = df.index
    df["cluster"] = adata_local.obs.loc[df.index, "domain"].astype(str)

    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["UMAP_1", "UMAP_2"], inplace=True)

    return df.reset_index(drop=True).to_dict(orient="records")

    
@app.get("/hvg-enrichment")   
def hvg_enrichment():
    """
    对adata.obs['domain']中的每个聚类执行功能富集分析
    
    返回:
    dict: 包含每个聚类的富集分析结果
    """
    global adata
    
    if "highly_variable" not in adata.var:
        return {"error": "Highly variable genes not computed."}
    
    # 获取所有聚类
    if "domain" not in adata.obs:
        return {"error": "Clustering results not found in adata.obs['domain']."}
    
    # clusters = adata.obs["domain"].unique()
    adata.obs["domain"] = adata.obs["domain"].astype(str).astype("category")
    clusters = adata.obs["domain"].cat.categories.tolist()
    
    organism = "Human"
    cutoff = 0.05
    
    # 获取可用的gene set
    available_sets = gp.get_library_name()
    print([s for s in available_sets if "WikiPathways" in s])
    
    # 明确指定多个gene set分类来源
    gene_sets = {
        "Biological Process": "GO_Biological_Process_2021",
        "Molecular Function": "GO_Molecular_Function_2021",
        "Cellular Component": "GO_Cellular_Component_2021",
        "WikiPathways": "WikiPathways_2024_Human",
        "Reactome": "Reactome_2022"
    }
    
    # 存储所有聚类的富集结果
    all_clusters_results = {}
    
    # 对每个聚类执行富集分析
    for cluster in clusters:
        print(f"Processing cluster: {cluster}")
        
        # 筛选当前聚类的细胞
        cluster_cells = adata.obs_names[adata.obs["domain"] == cluster]
        
        # 获取差异表达基因 (可选使用rank_genes_groups)
        cluster = str(cluster)
        sc.tl.rank_genes_groups(adata, groupby='domain', groups=[cluster], reference='rest', method='wilcoxon')
        top_genes = adata.uns['rank_genes_groups']['names'][cluster][:100].tolist()
        if not top_genes:
                    all_clusters_results[cluster] = {"error": f"No differentially expressed genes found for cluster {cluster}"}
                    continue
        
        all_results = []
        

        for category, gene_set in gene_sets.items():
            try:
                enr = gp.enrichr(
                    gene_list=top_genes,
                    gene_sets=gene_set,
                    organism=organism,
                    outdir=None,
                    cutoff=cutoff,
                )
                df = enr.results.copy()
                
                if not df.empty:
                    df["Gene_set"] = gene_set
                    df["Category"] = category  
                    all_results.append(df)
            except Exception as e:
                print(f"Failed for {cluster}, {gene_set}: {e}")
        
        if not all_results:
            all_clusters_results[cluster] = {"error": f"No enrichment results for cluster {cluster}"}
            continue
        
        merged_df = pd.concat(all_results)
        merged_df = merged_df.sort_values("Adjusted P-value")
        
        top_results = (
            merged_df.groupby("Category", group_keys=False)
            .apply(lambda x: x.head(8))
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        
        all_clusters_results[cluster] = top_results
    
    return all_clusters_results

@app.get("/hvg-enrichment-cluster") 
def hvg_enrichment_by_clusters(cluster:str = Query(...)):
    """
    对adata.obs['domain']中的每个聚类执行功能富集分析
    
    返回:
    dict: 包含每个聚类的富集分析结果
    """
    global adata
    
    if "highly_variable" not in adata.var:
        return {"error": "Highly variable genes not computed."}
    
    # 获取所有聚类
    if "domain" not in adata.obs:
        return {"error": "Clustering results not found in adata.obs['domain']."}
    
    # clusters = adata.obs["domain"].unique()
    adata.obs["domain"] = adata.obs["domain"].astype(str).astype("category")
    # clusters = adata.obs["domain"].cat.categories.tolist()
    
    organism = "Human"
    cutoff = 0.05
    
    # 获取可用的gene set
    available_sets = gp.get_library_name()
    print([s for s in available_sets if "WikiPathways" in s])
    
    # 明确指定多个gene set分类来源
    gene_sets = {
        "Biological Process": "GO_Biological_Process_2021",
        "Molecular Function": "GO_Molecular_Function_2021",
        "Cellular Component": "GO_Cellular_Component_2021",
        "WikiPathways": "WikiPathways_2024_Human",
        "Reactome": "Reactome_2022"
    }
    
    # 存储所有聚类的富集结果
    all_clusters_results = {}
    
    # 对每个聚类执行富集分析
    
    print(f"Processing cluster: {cluster}")
    
    # 筛选当前聚类的细胞
    cluster_cells = adata.obs_names[adata.obs["domain"] == cluster]
    
    # 获取差异表达基因 (可选使用rank_genes_groups)
    cluster = str(cluster)
    sc.tl.rank_genes_groups(adata, groupby='domain', groups=[cluster], reference='rest', method='wilcoxon')
    top_genes = adata.uns['rank_genes_groups']['names'][cluster][:100].tolist()
    if not top_genes:
                all_clusters_results[cluster] = {"error": f"No differentially expressed genes found for cluster {cluster}"}
                return
    
    all_results = []
    

    for category, gene_set in gene_sets.items():
        try:
            enr = gp.enrichr(
                gene_list=top_genes,
                gene_sets=gene_set,
                organism=organism,
                outdir=None,
                cutoff=cutoff,
            )
            df = enr.results.copy()
            
            if not df.empty:
                df["Gene_set"] = gene_set
                df["Category"] = category  
                all_results.append(df)
        except Exception as e:
            print(f"Failed for {cluster}, {gene_set}: {e}")
    
    if not all_results:
        all_clusters_results[cluster] = {"error": f"No enrichment results for cluster {cluster}"}
        return
    
    merged_df = pd.concat(all_results)
    merged_df = merged_df.sort_values("Adjusted P-value")
    
    top_results = (
        merged_df.groupby("Category", group_keys=False)
        .apply(lambda x: x.head(8))
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
    
    all_clusters_results[cluster] = top_results
    
    return all_clusters_results

@app.get("/cellchat")
def cell_chat():
    global adata
    print("Shape:", adata.shape)
    print("var_names:", adata.var_names.tolist()[:10])  # 只看前10个
    adata_copy = adata.copy()


    cluster_labels = adata_copy.obs["domain"]
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)


    spatial_coords = adata_copy.obsm['spatial']
    tri = Delaunay(spatial_coords)
    edges = set()
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edges.add((simplex[i], simplex[j]))
                edges.add((simplex[j], simplex[i]))  


    n_cells = adata_copy.n_obs
    adj_matrix = np.zeros((n_cells, n_cells), dtype=bool)
    for i, j in edges:
        adj_matrix[i, j] = True


    cluster_interaction_matrix = np.zeros((n_clusters, n_clusters))
    cluster_mapping = {cluster: idx for idx, cluster in enumerate(unique_clusters)}

    for i in range(n_cells):
        for j in range(n_cells):
            if adj_matrix[i, j]:
                source_cluster = cluster_labels[i]
                target_cluster = cluster_labels[j]
                source_idx = cluster_mapping[source_cluster]
                target_idx = cluster_mapping[target_cluster]
                cluster_interaction_matrix[source_idx, target_idx] += 1

    for i in range(n_clusters):
        source_cluster = unique_clusters[i]
        n_cells_in_source = np.sum(cluster_labels == source_cluster)
        if n_cells_in_source > 0:  # 避免除以零
            cluster_interaction_matrix[i, :] /= n_cells_in_source


    lr_pairs = [
        ('TGFB1', 'TGFBR1'), ('TGFB1', 'TGFBR2'),
        ('CD274', 'PDCD1'), ('PDCD1LG2', 'PDCD1'),
        ('CD40LG', 'CD40'), ('IL2', 'IL2RA'),
        ('IL7', 'IL7R'), ('IFNG', 'IFNGR1'),
        ('TNF', 'TNFRSF1A'), ('TNF', 'TNFRSF1B'),
        ('VEGFA', 'KDR'), ('VEGFA', 'FLT1')
    ]


    all_genes = set(adata_copy.var_names)
    valid_lr_pairs = []
    for ligand, receptor in lr_pairs:
        if ligand in all_genes and receptor in all_genes:
            valid_lr_pairs.append((ligand, receptor))

    if not valid_lr_pairs:
        print("没有找到匹配的配体-受体对，请更新列表或检查基因名称格式")
    else:

        lr_cluster_exp = {}
        for cluster in unique_clusters:
            cells_in_cluster = cluster_labels == cluster
            cluster_expr = adata_copy[cells_in_cluster].X
            

            if isinstance(cluster_expr, np.ndarray):
                pass
            else: 
                cluster_expr = cluster_expr.toarray()
            
            lr_cluster_exp[cluster] = {}
            for gene in set(sum(valid_lr_pairs, ())):
                if gene in all_genes:
                    gene_idx = list(adata_copy.var_names).index(gene)
                    lr_cluster_exp[cluster][gene] = np.mean(cluster_expr[:, gene_idx])

        lr_interaction_scores = []
        for source_cluster in unique_clusters:
            for target_cluster in unique_clusters:
                if source_cluster == target_cluster:
                    continue  
                    
                for ligand, receptor in valid_lr_pairs:
                    if ligand in lr_cluster_exp[source_cluster] and receptor in lr_cluster_exp[target_cluster]:
                        ligand_exp = lr_cluster_exp[source_cluster][ligand]
                        receptor_exp = lr_cluster_exp[target_cluster][receptor]
                        
                
                        interaction_score = ligand_exp * receptor_exp
                        
                        lr_interaction_scores.append({
                            'source_cluster': source_cluster,
                            'target_cluster': target_cluster,
                            'ligand': ligand,
                            'receptor': receptor,
                            'ligand_exp': ligand_exp,
                            'receptor_exp': receptor_exp,
                            'interaction_score': interaction_score
                        })


        lr_df = pd.DataFrame(lr_interaction_scores)
        lr_df = lr_df.sort_values('interaction_score', ascending=False)
        

        top_n = 30  
        if len(lr_df) > top_n:
            top_interactions = lr_df.head(top_n)
        else:
            top_interactions = lr_df

        cluster_pair_scores = lr_df.groupby(['source_cluster', 'target_cluster'])['interaction_score'].sum().reset_index()
        
    
        # lr_df.to_csv('ligand_receptor_interactions.csv', index=False)
        # cluster_pair_scores.to_csv('cluster_interaction_scores.csv', index=False)
        
        return top_interactions.to_dict(orient="records")

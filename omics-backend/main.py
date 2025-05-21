import os

os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import squidpy as sq
import scanpy as sc
import pandas as pd
import json
from fastapi.staticfiles import StaticFiles
from sqlalchemy import Table, Column, Integer, String, MetaData, TIMESTAMP, Float, func, create_engine, insert, text
from sqlalchemy.exc import ProgrammingError, IntegrityError
from rpy2.robjects import pandas2ri
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import cross_val_predict
import gseapy as gp
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
db = os.getenv("DB_NAME")



# 建立连接
# engine = create_engine("mysql+pymysql://root:@localhost/omics_data", echo=True)
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db}", echo=True)
metadata = MetaData()

pandas2ri.activate()


def insert_initial_clusters(adata, engine, slice_id):
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table_name = f"spot_cluster_{slice_id}"
    spot_cluster = metadata.tables[table_name]

    with engine.connect() as conn:
        # 1. 检查是否已有数据
        result = conn.execute(spot_cluster.select().limit(1)).fetchone()
        if result is not None:
            print(f"Table {table_name} already has data. Skipping insertion.")
            return  # ✅ 跳过插入
        
        # 2. 否则插入
        for i, (barcode, row) in enumerate(adata.obs.iterrows()):
            n_count = float(row["nCount_Spatial"]) if "nCount_Spatial" in row else None
            cluster = str(row["leiden"])
            n_feature = float(row.get("nFeature_Spatial", None))
            percent_mito = float(row.get("pct_counts_mt", None))
            percent_ribo = float(row.get("pct_counts_ribo", None))
            x, y = map(float, adata.obsm["spatial"][i])
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

# app.mount("/images", StaticFiles(directory="./data/151673/spatial"), name="images")

# 全局路径与缓存
slice_id = "151673"
path = f"./data/{slice_id}"
spatial_dir = os.path.join(path, "spatial")

adata = None
expression_data = None


@app.get("/images/{slice_id}/tissue_hires_image.png")
def get_image(slice_id: str):
    path = f"./data/{slice_id}/spatial/tissue_hires_image.png"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png")


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
    sc.tl.leiden(adata_local, key_added="leiden")
    
    adata_local.obs["leiden_original"] = adata_local.obs["leiden"].copy()
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

scale ="hires"
spatial_dir = os.path.join(f"./data/{slice_id}", "spatial")
with open(os.path.join(spatial_dir, "scalefactors_json.json"), "r") as f:
    sf = json.load(f)
scale_key = "tissue_hires_scalef" if scale == "hires" else "tissue_lowres_scalef"
factor = sf[scale_key]
    
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
    all_barcodes = adata.obs.index.tolist()
    selected_barcodes = req.barcode
    selected_barcodes = list(selected_barcodes)  # 确保是列表类型
    try:
        # 1. 从数据库加载当前的聚类标签
        table_name = f"spot_cluster_{req.slice_id}"
        with engine.connect() as conn:
            db_clusters = pd.read_sql(
                text(f"SELECT barcode, cluster FROM `{table_name}`"),
                conn,
            ).set_index("barcode")

        # 2. 提取特征
        all_features = extract_features(adata, use_hvg_only=True, n_pcs=30)

        # 3. 合并数据库标签
        all_features = all_features.join(db_clusters.rename(columns={"cluster": "cluster"}))
        if all_features["cluster"].isnull().any():
            print("⚠️ 有 barcode 在数据库中找不到匹配的 cluster")
        
        
        # # 提取所有spot的特征
        # all_features = extract_features(adata, use_hvg_only=True, n_pcs=30)
        # # print("all_features",all_features)
        
        # # 添加聚类标签
        # all_features['cluster'] = adata.obs['leiden_original']
        
        # 分离训练集和预测集
        train_mask = ~all_features.index.isin(selected_barcodes)
        predict_mask = all_features.index.isin(selected_barcodes)
        
        train_data = all_features[train_mask].copy()
        predict_data = all_features[predict_mask].copy()
        
        
        # 检查是否有正确筛选出所有选中的barcode
        missing_barcodes = [b for b in selected_barcodes if b not in predict_data.index]
        if missing_barcodes:
            print(f"警告: {len(missing_barcodes)} 个选中的barcode在预测数据中缺失")
            print(f"缺失的barcode: {missing_barcodes[:5]}...")
        
        # 保存原始cluster信息以便后续比较
        predict_data_original = predict_data[['cluster']].copy()
        
        # 确保预测数据与选中的barcode一致
        if len(predict_data) != len(selected_barcodes):
            # 可能有barcode不在原始数据中，重新确认实际用于预测的barcode
            selected_barcodes = list(predict_data.index)
        
        # 准备特征和标签
        feature_cols = [col for col in train_data.columns if col != 'cluster']
        X_train = train_data[feature_cols]
        y_train = train_data['cluster']
        X_predict = predict_data[feature_cols]
        
        # 训练随机森林模型
        print("\n训练随机森林模型...")
        param_grid = {
            'n_estimators': [100],
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
        
        # print("\nTop 10 最重要特征:")
        # print(feature_importance.head(10))
        
        # 预测选中spot的新聚类
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
        
        print("all cluster_change_df", cluster_change_df)
        
        # 添加变化状态列
        cluster_change_df['changed'] = cluster_change_df['original_cluster'] != cluster_change_df['new_cluster']
        
        # 添加p值列 (1 - 置信度可以近似为p值)
        cluster_change_df['p_value'] = 1 - cluster_change_df['confidence']
        # 使用交叉验证获得训练集上的置信度分布
        cv_probs = cross_val_predict(best_rf, X_train, y_train, method='predict_proba', cv=5)
        cv_max_probs = np.max(cv_probs, axis=1)
        
        # 对每个预测，计算相对于训练集分布的p值
        p_values_refined = []
        
        # 确保p_values_refined与cluster_change_df长度一致
        for i, idx in enumerate(cluster_change_df.index):
            confidence = cluster_change_df.loc[idx, 'confidence']
            p_value = (cv_max_probs <= confidence).mean()
            p_values_refined.append(p_value)
        
        
        # 更新p值
        cluster_change_df['p_value_refined'] = p_values_refined
        
        # 计算第二高概率及其对应的类别
        second_probs = []
        second_classes = []
        diff_to_top = []  # 记录最高概率与第二高概率的差距
        
        for i, idx in enumerate(cluster_change_df.index):
            row = prediction_probs[i]
            sorted_idx = np.argsort(row)[::-1]  # 降序排列的索引
            top_class_idx = sorted_idx[0]
            second_class_idx = sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]
            
            top_prob = row[top_class_idx]
            second_prob = row[second_class_idx]
            
            second_probs.append(second_prob)
            second_classes.append(class_names[second_class_idx])
            diff_to_top.append(top_prob - second_prob)
        
        # 将第二高概率信息添加到结果中
        cluster_change_df['second_prob'] = second_probs
        cluster_change_df['second_class'] = second_classes
        cluster_change_df['prob_diff'] = diff_to_top
        
        # 更新adata中的聚类标签
        # 创建一个新的Series来存储预测的聚类结果，默认使用原始聚类
        predicted_clusters = pd.Series(adata.obs['leiden'].values, index=adata.obs.index)
        
        # 更新选中点的聚类标签
        for idx, row in cluster_change_df.iterrows():
            if idx in predicted_clusters.index:  # 确保索引存在
                predicted_clusters[idx] = row['new_cluster']
        
        # 将预测的聚类结果保存到adata
        adata.obs['predicted_cluster'] = predicted_clusters
        adata.obs['predicted_cluster'] = adata.obs['predicted_cluster'].astype('category')
        
        # 直接在adata.obs中创建cluster_changed列
        adata.obs['cluster_changed'] = adata.obs['leiden'] != adata.obs['predicted_cluster']
        # print(f"cluster_changed统计: {adata.obs['cluster_changed'].value_counts()}")
        
        # 检查变化情况，确保正确反映cluster变化
        selected_changed = adata.obs.loc[selected_barcodes, 'cluster_changed']
        # print(f"选中的spot中变化的数量: {selected_changed.sum()} (共 {len(selected_changed)})")
        
        # Debug: 检查选中spot的新旧类别
        for idx in selected_barcodes[:5]:  # 只检查前5个
            old = adata.obs.loc[idx, 'leiden']
            new = adata.obs.loc[idx, 'predicted_cluster']
            is_changed = old != new
            # print(f"Spot {idx}: 原始={old}, 预测={new}, 是否变化={is_changed}")
        
        # 输出变化的总结
        changed_spots = cluster_change_df[cluster_change_df['changed']]
        unchanged_spots = cluster_change_df[~cluster_change_df['changed']]
        
        print(f"\n总共有 {len(changed_spots)} 个spot的cluster发生了变化 (共{len(cluster_change_df)}个)")
        
        # 变化spot的置信度分析
        if len(changed_spots) > 0:
           
            
            # 置信度分布
            conf_bins = [0, 0.5, 0.7, 0.8, 0.9, 1.0]
            conf_labels = ['0-0.5', '0.5-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
            conf_counts = pd.cut(changed_spots['confidence'], bins=conf_bins).value_counts().sort_index()
            
            # print("\n变化spot的置信度分布:")
            for i, (level, count) in enumerate(zip(conf_labels, conf_counts)):
                pct = count / len(changed_spots) * 100
                # print(f"{level}: {count} ({pct:.1f}%)")
        
        # 创建DataFrame保存所有预测细节
        prediction_details = pd.DataFrame(index=adata.obs.index)
        prediction_details['is_selected'] = adata.obs.index.isin(selected_barcodes)
        prediction_details['original_cluster'] = adata.obs['leiden']
        prediction_details['predicted_cluster'] = adata.obs['predicted_cluster']
        
        # 为选定的barcode填充更详细的信息
        prediction_details['confidence'] = np.nan
        prediction_details['p_value'] = np.nan
        prediction_details['p_value_refined'] = np.nan
        prediction_details['cluster_changed'] = False
        prediction_details['second_prob'] = np.nan
        prediction_details['second_class'] = "NA"
        prediction_details['prob_diff'] = np.nan
        
        # 添加所有类别的概率列
        for cls in class_names:
            prob_col = f'prob_{cls}'
            prediction_details[prob_col] = np.nan
        
        # 填充选定的barcode信息
        for idx, row in cluster_change_df.iterrows():
            if idx in prediction_details.index:  # 确保索引存在
                prediction_details.loc[idx, 'confidence'] = float(row['confidence'])
                prediction_details.loc[idx, 'p_value'] = float(row['p_value'])
                prediction_details.loc[idx, 'p_value_refined'] = float(row['p_value_refined'])
                prediction_details.loc[idx, 'cluster_changed'] = bool(row['changed'])
                prediction_details.loc[idx, 'second_prob'] = float(row['second_prob'])
                prediction_details.loc[idx, 'second_class'] = str(row['second_class'])
                prediction_details.loc[idx, 'prob_diff'] = float(row['prob_diff'])
                
                # 填充所有类别的概率
                for cls in class_names:
                    prob_col = f'prob_{cls}'
                    if prob_col in row:
                        prediction_details.loc[idx, prob_col] = float(row[prob_col])
        
        # 将prediction_details中的所有列添加到adata.obs
        print("将prediction_details的所有列添加到adata.obs...")
        for col in prediction_details.columns:
            col_name = f"pred_{col}"
            if prediction_details[col].dtype == bool:
                adata.obs[col_name] = prediction_details[col].astype(bool)
            elif pd.api.types.is_numeric_dtype(prediction_details[col]):
                adata.obs[col_name] = prediction_details[col].astype(float)
            else:
                adata.obs[col_name] = prediction_details[col].astype(str)
        
        # 保存prediction_details到CSV文件
        prediction_details.to_csv(f"{slice_id}_prediction_details.csv")
        cluster_change_df.to_csv(f"{slice_id}_changed_details.csv")
        print(f"预测详情已保存到 {slice_id}_prediction_details.csv")
        
        # 根据置信度对变化的spots进行分类
        changed_mask = prediction_details['cluster_changed'] == True
        if changed_mask.any():
            conf_categories = pd.cut(
                prediction_details.loc[changed_mask, 'confidence'],
                bins=[0, 0.5, 0.7, 0.8, 0.9, 1.0],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
            
            if not conf_categories.empty:
                # 初始化为NA
                prediction_details['confidence_category'] = "NA"
                # 只更新变化的spot
                prediction_details.loc[changed_mask, 'confidence_category'] = conf_categories
                # 添加到adata
                adata.obs['pred_confidence_category'] = prediction_details['confidence_category'].astype(str)
                
                # 添加可靠性评估
                reliability = pd.Series("NA", index=prediction_details.index)
                
                # 定义置信度阈值和对应的可靠性标签
                confidence_thresholds = [(0.9, 1.0, "highly_reliable"),
                                    (0.8, 0.9, "reliable"),
                                    (0.7, 0.8, "moderately_reliable"),
                                    (0.5, 0.7, "questionable"),
                                    (0.0, 0.5, "unreliable")]
                
                # 只更新变化的spot的可靠性
                for min_conf, max_conf, label in confidence_thresholds:
                    mask = changed_mask & (prediction_details['confidence'] >= min_conf) & (prediction_details['confidence'] < max_conf)
                    reliability[mask] = label
                
                # 添加到adata
                adata.obs['pred_reliability'] = reliability.astype(str)
                
                # 特别标记高可靠性的变化spot
                high_confidence_change = prediction_details['cluster_changed'] == True
                high_confidence_change = high_confidence_change & (prediction_details['confidence'] >= 0.8)
                adata.obs['pred_high_confidence_change'] = high_confidence_change.astype(bool)
        
        # 添加更多的可视化
        # 可视化原始聚类结果
       
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
                        if 'spatial' in key or 'leiden' in key:
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

    if "X_umap" not in adata.obsm:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
        adata = adata[:, adata.var.highly_variable]
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    
    df = pd.DataFrame(
        adata.obsm["X_umap"],
        index=adata.obs_names,  
        columns=["UMAP_1", "UMAP_2"]
    )

    df["barcode"] = df.index
    df["cluster"] = adata.obs.loc[df.index, "leiden"].astype(str)

    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["UMAP_1", "UMAP_2"], inplace=True)

    return df.reset_index(drop=True).to_dict(orient="records")

class EnrichmentRequest(BaseModel):
    organism: Optional[str] = "Human"
    gene_sets: Optional[List[str]] = ["GO_Biological_Process_2021"]
    cutoff: Optional[float] = 0.05

@app.post("/hvg-enrichment")
def hvg_enrichment(request: EnrichmentRequest):
    if "highly_variable" not in adata.var:
        return {"error": "Highly variable genes not computed."}

    hvg_genes = adata.var_names[adata.var["highly_variable"]].tolist()
    if not hvg_genes:
        return {"error": "No highly variable genes found."}

    all_results = []

    for gene_set in request.gene_sets:
        try:
            enr = gp.enrichr(
                gene_list=hvg_genes,
                gene_sets=gene_set,
                organism=request.organism,
                outdir=None,
                cutoff=request.cutoff
            )
            df = enr.results.copy()
            df["Gene_set"] = gene_set  # 添加来源标记
            all_results.append(df)
        except Exception as e:
            continue  # 可以选择记录或跳过失败的 gene_set

    if not all_results:
        return {"error": "No enrichment results."}

    merged_df = pd.concat(all_results).sort_values("Adjusted P-value")
    top_results = merged_df.head(40).to_dict(orient="records")

    return {"results": top_results}
    
    
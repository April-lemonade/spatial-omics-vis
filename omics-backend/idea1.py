import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.model_selection import cross_val_predict
import os

# 1. 读取数据
section_id = '151673'
file_fold = '/home/junning/projectnvme/visulization/' + str(section_id)
adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
adata.var_names_make_unique()

print(f"加载的数据形状: {adata.shape}")
print(f"可用的观测值数量: {adata.n_obs}")
print(f"可用的基因数量: {adata.n_vars}")

# 2. 对空间转录组数据进行预处理和聚类
# 预处理
print("进行数据预处理...")
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=2000)

# PCA降维
sc.pp.pca(adata, n_comps=30, use_highly_variable=True, svd_solver='arpack')

# 计算邻居图
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)

# 使用Leiden算法进行聚类
print("使用Leiden算法进行聚类...")
sc.tl.leiden(adata, resolution=0.5)

# 保存初始聚类结果
adata.obs['initial_cluster'] = adata.obs['leiden']

print(f"总共找到 {len(adata.obs['leiden'].unique())} 个聚类")
print(adata.obs['leiden'].value_counts())

# UMAP可视化 (可选)
sc.tl.umap(adata)
sc.pl.umap(adata, color='leiden', save=f'_{section_id}_clusters.pdf')
sc.pl.spatial(adata, color='leiden', save=f'_{section_id}_spatial_clusters.pdf')

# 3. 随机选择一些spot进行重分类（模拟标记不准确的spot）
# 随机选择20个spot，或者按真实需求选择有问题的spot
n_samples = 20  # 可以调整这个数字
np.random.seed(42)  # 设置随机种子以保证结果可重复
all_barcodes = adata.obs.index.tolist()
selected_barcodes = np.random.choice(all_barcodes, size=n_samples, replace=False)
selected_barcodes = list(selected_barcodes)  # 确保是列表类型

print(f"随机选择了 {len(selected_barcodes)} 个spot进行重分类")
print(f"选中的前5个barcode: {selected_barcodes[:5]}")

# 4. 提取更有意义的特征，并进行适当的标准化
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

# 5. 提取所有需要的特征
print("提取特征中...")
try:
    # 提取所有spot的特征
    all_features = extract_features(adata, use_hvg_only=True, n_pcs=30)
    print("all_features",all_features)
    # 添加聚类标签
    all_features['cluster'] = adata.obs['leiden']
    
    # 分离训练集和预测集
    train_mask = ~all_features.index.isin(selected_barcodes)
    predict_mask = all_features.index.isin(selected_barcodes)
    
    train_data = all_features[train_mask].copy()
    predict_data = all_features[predict_mask].copy()
    
    print(f"训练数据形状: {train_data.shape}, 预测数据形状: {predict_data.shape}")
    print(f"预测数据的索引长度: {len(predict_data.index)}")
    print(f"预测数据的索引是否唯一: {len(predict_data.index) == len(predict_data.index.unique())}")
    
    # 检查是否有正确筛选出所有选中的barcode
    missing_barcodes = [b for b in selected_barcodes if b not in predict_data.index]
    if missing_barcodes:
        print(f"警告: {len(missing_barcodes)} 个选中的barcode在预测数据中缺失")
        print(f"缺失的barcode: {missing_barcodes[:5]}...")
    
    # 保存原始cluster信息以便后续比较
    predict_data_original = predict_data[['cluster']].copy()
    
    # 确保预测数据与选中的barcode一致
    if len(predict_data) != len(selected_barcodes):
        print(f"警告: 预测数据长度 ({len(predict_data)}) 与选中barcode数量 ({len(selected_barcodes)}) 不匹配")
        # 可能有barcode不在原始数据中，重新确认实际用于预测的barcode
        selected_barcodes = list(predict_data.index)
        print(f"更新后的选中barcode数量: {len(selected_barcodes)}")
    
    # 准备特征和标签
    feature_cols = [col for col in train_data.columns if col != 'cluster']
    X_train = train_data[feature_cols]
    y_train = train_data['cluster']
    X_predict = predict_data[feature_cols]
    
    print(f"训练集形状: {X_train.shape}, 预测集形状: {X_predict.shape}")
    print(f"特征列表: {feature_cols}")
    
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
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"交叉验证最佳得分: {grid_search.best_score_:.4f}")
    
    # 识别最重要的特征
    importances = best_rf.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 最重要特征:")
    print(feature_importance.head(10))
    
    # 预测选中spot的新聚类
    new_clusters = best_rf.predict(X_predict)
    
    # 获取预测的概率值（所有类别的概率分布）
    prediction_probs = best_rf.predict_proba(X_predict)
    print("all prediction_probs",prediction_probs)
    print(f"预测概率矩阵形状: {prediction_probs.shape}")
    # 获取每个预测的置信度 (最高概率值)
    confidence_scores = np.max(prediction_probs, axis=1)
    print(f"选定barcode数量: {len(selected_barcodes)}, 预测数据形状: {X_predict.shape}, 置信度长度: {len(confidence_scores)}")
    
    # 创建概率分布DataFrame，包含所有类别的概率
    class_names = best_rf.classes_
    prob_cols = [f"prob_{cls}" for cls in class_names]
    probs_df = pd.DataFrame(prediction_probs, columns=prob_cols, index=predict_data.index)
    print("all probs_df",probs_df)
    print(f"probs_df形状: {probs_df.shape}")
    
    # 创建结果跟踪DataFrame - 确保使用预测数据的索引
    result_index = predict_data.index
    cluster_change_df = pd.DataFrame(index=result_index)
    cluster_change_df['barcode'] = result_index
    cluster_change_df['original_cluster'] = predict_data_original['cluster'].values
    cluster_change_df['new_cluster'] = new_clusters
    cluster_change_df['confidence'] = confidence_scores
    
    print(f"cluster_change_df形状: {cluster_change_df.shape}")
    print(f"cluster_change_df索引长度: {len(cluster_change_df.index)}")
    
    # 将概率分布添加到结果中 - 使用索引合并而不是concat
    for col in probs_df.columns:
        cluster_change_df[col] = probs_df[col]
    
    print("all cluster_change_df", cluster_change_df)
    
    # 添加变化状态列
    cluster_change_df['changed'] = cluster_change_df['original_cluster'] != cluster_change_df['new_cluster']
    
    # 添加p值列 (1 - 置信度可以近似为p值)
    cluster_change_df['p_value'] = 1 - cluster_change_df['confidence']
    print("all cluster_change_df", cluster_change_df)
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
    
    print(f"p_values_refined长度: {len(p_values_refined)}, cluster_change_df长度: {len(cluster_change_df)}")
    
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
    print(f"cluster_changed统计: {adata.obs['cluster_changed'].value_counts()}")
    
    # 检查变化情况，确保正确反映cluster变化
    selected_changed = adata.obs.loc[selected_barcodes, 'cluster_changed']
    print(f"选中的spot中变化的数量: {selected_changed.sum()} (共 {len(selected_changed)})")
    
    # Debug: 检查选中spot的新旧类别
    for idx in selected_barcodes[:5]:  # 只检查前5个
        old = adata.obs.loc[idx, 'leiden']
        new = adata.obs.loc[idx, 'predicted_cluster']
        is_changed = old != new
        print(f"Spot {idx}: 原始={old}, 预测={new}, 是否变化={is_changed}")
    
    # 输出变化的总结
    changed_spots = cluster_change_df[cluster_change_df['changed']]
    unchanged_spots = cluster_change_df[~cluster_change_df['changed']]
    
    print(f"\n总共有 {len(changed_spots)} 个spot的cluster发生了变化 (共{len(cluster_change_df)}个)")
    
    # 变化spot的置信度分析
    if len(changed_spots) > 0:
        print("\n变化spot的置信度分析:")
        print(f"置信度均值: {changed_spots['confidence'].mean():.4f}")
        print(f"置信度中位数: {changed_spots['confidence'].median():.4f}")
        print(f"置信度最小值: {changed_spots['confidence'].min():.4f}")
        print(f"置信度最大值: {changed_spots['confidence'].max():.4f}")
        print(f"与第二高概率的平均差距: {changed_spots['prob_diff'].mean():.4f}")
        
        # 置信度分布
        conf_bins = [0, 0.5, 0.7, 0.8, 0.9, 1.0]
        conf_labels = ['0-0.5', '0.5-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        conf_counts = pd.cut(changed_spots['confidence'], bins=conf_bins).value_counts().sort_index()
        
        print("\n变化spot的置信度分布:")
        for i, (level, count) in enumerate(zip(conf_labels, conf_counts)):
            pct = count / len(changed_spots) * 100
            print(f"{level}: {count} ({pct:.1f}%)")
    
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
    prediction_details.to_csv(f"{section_id}_prediction_details.csv")
    cluster_change_df.to_csv(f"{section_id}_changed_details.csv")
    print(f"预测详情已保存到 {section_id}_prediction_details.csv")
    
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
    sc.pl.spatial(adata, color='leiden', title='原始聚类结果', save=f'_{section_id}_original_clusters.pdf')
    
    # 可视化预测聚类结果 
    sc.pl.spatial(adata, color='predicted_cluster', title='预测聚类结果', save=f'_{section_id}_predicted_clusters.pdf')
    
    # 对变化的spot进行可视化
    # 注意：我们已经在前面创建了adata.obs['cluster_changed']列
    sc.pl.spatial(adata, color='cluster_changed', title='变化的spot', save=f'_{section_id}_changed_spots.pdf')
    
    # 对置信度进行可视化
    sc.pl.spatial(adata, color='pred_confidence', title='预测置信度', save=f'_{section_id}_confidence.pdf')
    adata.uns["change_df"] = cluster_change_df
    # 可视化高置信度变化的spots
    if 'pred_high_confidence_change' in adata.obs:
        sc.pl.spatial(adata, color='pred_high_confidence_change', 
                      title='高置信度变化的spot', 
                      save=f'_{section_id}_high_confidence_changes.pdf')
    
    # 可视化可靠性分类
    if 'pred_reliability' in adata.obs:
        sc.pl.spatial(adata, color='pred_reliability', 
                     title='预测可靠性分类', 
                     save=f'_{section_id}_reliability.pdf')
    
    # 确保目录存在
    os.makedirs('visulization', exist_ok=True)
    
    # 保存结果
    try:
        output_path = f"{section_id}_processed.h5ad"  # 直接保存在当前目录
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
            output_path = f"{section_id}_processed_backup.h5ad"
            adata_backup.write_h5ad(output_path)
            print(f"成功保存到 {output_path}")
            
            # 再次尝试保存完整数据为其他格式
            adata.write_loom(f"{section_id}_complete.loom")
            print(f"完整数据已保存为 {section_id}_complete.loom")
        except Exception as e2:
            print(f"备用保存方法也失败: {str(e2)}")
            # 最后尝试保存关键信息
            obs_df = adata.obs.copy()
            obs_df.to_csv(f"{section_id}_obs_data.csv")
            print(f"已将观测数据保存到 {section_id}_obs_data.csv")
    
    # 输出统计摘要
    summary_stats = {
        'total_selected': len(selected_barcodes),
        'total_changed': len(changed_spots),
        'percent_changed': len(changed_spots) / len(selected_barcodes) * 100 if len(selected_barcodes) > 0 else 0,
        'avg_confidence_changed': changed_spots['confidence'].mean() if len(changed_spots) > 0 else None,
        'avg_confidence_unchanged': unchanged_spots['confidence'].mean() if len(unchanged_spots) > 0 else None,
        'avg_p_value': changed_spots['p_value'].mean() if len(changed_spots) > 0 else None,
        'avg_p_value_refined': changed_spots['p_value_refined'].mean() if len(changed_spots) > 0 else None
    }
    
    print("\n变化统计摘要:")
    for key, value in summary_stats.items():
        print(f"{key}: {value}")
    

except Exception as e:
    print(f"处理过程中出错: {str(e)}")
    import traceback
    traceback.print_exc()


## 🛠️ 环境准备

项目使用 [Poetry](https://python-poetry.org/) 管理 Python 依赖。

### 1. 安装 Poetry（如未安装）

```bash
curl -sSL https://install.python-poetry.org | python3 -
```
### 2. 安装依赖
```bash
poetry install
```
### 3. 启动虚拟环境
```bash
poetry shell
```
### 4. 导入文件到data目录
```angular2html
data/
└── 151673/
    ├── filtered_feature_bc_matrix.h5
    ├── info.json
    ├── metadata.tsv
    ├── scRNA.h5ad
    ├── info.json
    └── spatial/
        ├── full_image.tif
        ├── scalefactors_json.json
        ├── tissue_hires_image.png
        ├── tissue_lowres_image.png
        └── tissue_positions_list.csv
```
其中，info.json 存储基本信息
示例
```json
{
  "tissue": "Human DLPFC (dorsolateral prefrontal cortex)",
  "platform": "10x Genomics Visium",
  "slice_id": "151673",
  "spot_diameter_um": 55,
  "spot_spacing_um": 100
}
```
### 数据库配置
本项目默认使用 MySQL 数据库，连接信息在代码中通过 SQLAlchemy 设置，在后端主目录添加.env文件连接数据库
```json
DB_USER=YOUR_DB_USER
DB_PASSWORD=YOUR_DB_PASSWORD
DB_HOST=YOUR_DB_HOST
DB_NAME=YOUR_DB_NAME
```

### 🚀 运行 main.py
```bash
fastapi dev main.py
```
接口文档：http://localhost:8000/docs
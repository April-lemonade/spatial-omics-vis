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
    └── spatial/
        ├── full_image.tif
        ├── scalefactors_json.json
        ├── tissue_hires_image.png
        ├── tissue_lowres_image.png
        └── tissue_positions_list.csv
```
### 数据库配置
本项目默认使用 MySQL 数据库，连接信息在代码中通过 SQLAlchemy 设置：
`engine = create_engine("mysql+pymysql://root:@localhost/omics_data", echo=True)`

 🗄️ 数据库连接参数（默认）

| 参数     | 默认值        |
|----------|---------------|
| 用户名   | `root`        |
| 密码     | （空）        |
| 主机     | `localhost`   |
| 数据库名 | `omics_data`  |

请确保本地 MySQL 服务已启动，并且存在名为 omics_data 的数据库。

### 🚀 运行 main.py
```bash
fastapi dev main.py
```
接口文档：http://localhost:8000/docs
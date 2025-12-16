# S-H-ESD 时间序列异常检测 MCP 服务器

作者：庞力铖

邮箱：3522236586@qq.com

GitHub:

基于季节性混合 ESD (Seasonal Hybrid ESD) 方法的时间序列异常检测 MCP 工具服务器。

## 功能特性

- 支持 CSV 和 TXT 格式的时间序列数据文件
- 自动推断季节性周期
- 使用 STL 分解和 GESD 算法进行异常检测
- 提供详细的检测结果，包括异常索引、时间、分数等
- **智能检测工具**：自动转换数据文件（如果需要）然后进行检测，避免"已转换"误报

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
python mcp_server.py
```

服务启动后将监听 `127.0.0.1:2701`，使用 SSE 传输方式。SSE 端点为：`http://127.0.0.1:2701/sse`

## 工具说明

### detect_with_auto_convert（推荐）

智能异常检测工具：自动转换数据文件（如果需要）然后进行异常检测。这是**推荐使用的工具**，它会自动处理转换和检测流程，无需手动操作。

#### 参数

- **data_file** (str, 必填): data文件夹中的文件路径（如 `data/HDFS_2k.log_structured.csv` 或 `HDFS_2k.log_structured.csv`）
- **seasonality** (int, 可选): 季节性周期长度（可选，默认自动推断）
- **alpha** (float, 可选): 显著性水平（默认 0.05）
- **max_anomalies** (int, 可选): 最大检测异常数（可选，默认自动）
- **aggregation_window** (str, 可选): 聚合窗口（仅用于日志文件，默认：'1min'。系统会自动根据数据时间跨度调整窗口大小，确保有足够的数据点）
- **log_mode** (bool, 可选): 是否强制使用日志模式（默认：False，自动检测）

#### 使用示例

```python
# 直接检测data文件夹中的文件，自动转换（如果需要）并检测
result = detect_with_auto_convert(data_file="data/HDFS_2k.log_structured.csv")

# 或者使用相对路径
result = detect_with_auto_convert(data_file="HDFS_2k.log_structured.csv")

# 检测结果包含转换信息和检测结果
print(result["conversion_result"])  # 转换信息
print(result["detection_result"])    # 检测结果
```

#### 工作流程

1. 检查 `models` 文件夹中是否已有转换后的文件
2. 如果没有，自动从 `data` 文件夹读取并转换
3. 使用转换后的文件进行异常检测
4. 返回转换和检测的完整结果

### list_data_files

列出 `data` 文件夹中的所有数据文件，并显示哪些文件已经转换。

#### 返回

包含文件列表的字典，每个文件包含：

- `name`: 文件名
- `path`: 完整路径
- `relative_path`: 相对路径
- `size`: 文件大小
- `converted`: 是否已转换

### auto_convert_data

自动转换 `data` 文件夹中的数据文件到 `models` 文件夹（CSV格式）。如果不指定文件名，则转换所有未转换的文件。

#### 参数

- **file_name** (str, 可选): 要转换的文件名（默认：转换所有未转换的文件）
- **aggregation_window** (str, 可选): 聚合窗口（仅用于日志文件，默认：'1min'。系统会自动根据数据时间跨度调整窗口大小，确保有足够的数据点）
- **log_mode** (bool, 可选): 是否强制使用日志模式（默认：False，自动检测）

#### 使用示例

```python
# 列出所有数据文件
files = list_data_files()

# 自动转换所有未转换的文件
result = auto_convert_data()

# 转换指定文件
result = auto_convert_data(file_name="HDFS_2k.log_structured.csv")
```

### convert_data_to_timeseries

将不同格式的数据文件转换为统一的时间序列格式（CSV格式），保存在 `models` 文件夹中。

#### 参数

- **input_file** (str, 必填): 输入文件路径（CSV、TXT或日志文件）
- **output_file** (str, 可选): 输出文件路径（默认：models/输入文件名.csv）
- **time_col** (str, 可选): 时间列名（仅用于标准CSV/TXT，默认：'time'）
- **value_col** (str, 可选): 数值列名（仅用于标准CSV/TXT，默认：'value'）
- **delimiter** (str, 可选): 分隔符（仅用于CSV/TXT，默认：','）
- **aggregation_window** (str, 可选): 聚合窗口（仅用于日志文件，如'1min', '5min', '1H'，默认：'1min'。系统会自动根据数据时间跨度调整窗口大小，确保有足够的数据点）
- **log_mode** (bool, 可选): 是否强制使用日志模式（按时间聚合日志数量，默认：False，自动检测）

#### 使用示例

```python
# 转换日志文件（自动检测格式）
result = convert_data_to_timeseries(
    input_file="data/HDFS_2k.log_structured.csv"
)
# 输出文件：models/HDFS_2k.csv

# 转换标准CSV文件
result = convert_data_to_timeseries(
    input_file="data/example.csv",
    time_col="timestamp",
    value_col="temperature"
)

# 转换日志文件并指定聚合窗口
result = convert_data_to_timeseries(
    input_file="data/OpenSSH_2k.log_structured.csv",
    aggregation_window="5min"
)
```

#### 命令行使用

```bash
# 转换日志文件
python data_converter.py data/HDFS_2k.log_structured.csv -o models/HDFS_2k.csv

# 转换标准CSV文件
python data_converter.py data/example.csv -o models/example.csv --time-col timestamp --value-col value
```

### detect_sh_esd_anomaly

使用季节性混合 ESD 方法检测时间序列异常。

#### 参数

- **file_path** (str, 必填): 数据文件路径，支持 CSV、TXT 和 JSON 格式（UTF-8 编码）
- **seasonality** (int, 可选): 季节性周期长度，默认自动推断
- **alpha** (float, 可选): 显著性水平，默认 0.05
- **max_anomalies** (int, 可选): 最大检测异常数，默认自动计算
- **format_opts** (dict, 可选): 格式选项，可包含：
  - `time_col`: 时间列名（默认 'time'）
  - `value_col`: 数值列名（默认 'value'）
  - `delimiter`: 分隔符（CSV 默认 ','，TXT 默认 '\t'）

#### 输入文件要求

- **CSV 文件**: 必须包含时间列和数值列，分隔符默认为逗号
- **TXT 文件**: 必须为表格形式，分隔符默认为制表符
- **时间格式**: 支持 ISO8601 字符串或 UNIX 时间戳（秒/毫秒），将统一转换为 ISO8601 格式
- **数值格式**: 必须为可转换为 float 的数值，无法转换的行将被丢弃

#### 输出格式

返回一个字典，包含以下字段：

```python
{
    "is_anomalous": bool,           # 是否检测到异常
    "n": int,                        # 有效数据点数量
    "seasonality": int,              # 使用的季节性周期
    "alpha": float,                  # 显著性水平
    "max_anomalies": int | "auto",   # 最大异常数
    "k_found": int,                  # 实际检测到的异常数量
    "iqr_lower": float,              # IQR 下界
    "iqr_upper": float,              # IQR 上界
    "anomaly_indices": list[int],    # 异常点的索引列表
    "anomaly_times": list[str],      # 异常点的时间列表（ISO8601 格式）
    "scores": list[float],           # 异常分数列表（0-1，越高越异常）
    "zscores": list[float],          # z-score 列表
    "dropped": int,                  # 丢弃的无效行数
    "message": str                   # 处理消息
}
```

## 使用示例

### 1、直接使用CSV文件

```python
# 假设有一个包含 'time' 和 'value' 列的 CSV 文件
result = detect_sh_esd_anomaly(
    file_path="data/example.csv"
)
```

### 2、指定列名

```python
# 如果时间列名为 'timestamp'，数值列名为 'temperature'
result = detect_sh_esd_anomaly(
    file_path="data/sensor_data.csv",
    format_opts={
        "time_col": "timestamp",
        "value_col": "temperature"
    }
)
```

### 3、自定义参数

```python
result = detect_sh_esd_anomaly(
    file_path="data/example.csv",
    seasonality=24,        # 指定周期为 24（小时数据，日周期）
    alpha=0.01,            # 更严格的显著性水平
    max_anomalies=10       # 最多检测 10 个异常
)
```

### 4、TXT 文件

```python
result = detect_sh_esd_anomaly(
    file_path="data/example.txt",
    format_opts={
        "delimiter": "\t",  # 制表符分隔
        "time_col": "date",
        "value_col": "count"
    }
)
```

## 算法说明

### S-H-ESD 方法

1. **数据预处理**: 读取文件并转换为统一格式 `list[{"time": str, "value": float}]`，按时间排序
2. **季节性分解**: 使用 STL (Seasonal and Trend decomposition using Loess) 将序列分解为：
   - `seasonal`: 季节性成分
   - `trend`: 趋势成分
   - `residual`: 残差成分
3. **异常检测**: 对残差部分应用 GESD (Generalized Extreme Studentized Deviate) 算法：
   - 迭代移除最极端的残差值
   - 计算 R_i 统计量和临界值 lambda_i（基于 t 分布）
   - 使用显著性水平 alpha 控制检测严格程度
4. **异常评分**: 基于残差的绝对 z-score，归一化到 [0,1] 区间

### 自动周期推断

如果未指定 `seasonality`，系统会自动推断：

- 默认使用 `n // 4`（n 为数据点数量）
- 最小周期为 2，最大周期为 `n // 2`
- 如果数据点少于 `2 * seasonality`，将使用简单的 z-score 方法

## 测试数据

项目包含 `data/` 文件夹，其中有 3 个示例数据集：

- `HDFS_2k.log_structured.csv`
- `OpenSSH_2k.log_structured.csv`
- `Spark_2k.log_structured.csv`

这些文件可用于测试和验证功能。

## 使用流程

**直接使用 `detect_with_auto_convert` 工具，它会自动处理所有步骤：**

```python
# 检测第一个数据集
result1 = detect_with_auto_convert(data_file="data/HDFS_2k.log_structured.csv")

# 检测第二个数据集（自动转换，如果尚未转换）
result2 = detect_with_auto_convert(data_file="data/OpenSSH_2k.log_structured.csv")

# 检测第三个数据集（自动转换，如果尚未转换）
result3 = detect_with_auto_convert(data_file="data/Spark_2k.log_structured.csv")
```

## 注意事项

1. 数据文件必须是 UTF-8 编码
2. 至少需要 3 个有效数据点才能进行检测
3. 进行季节性分解需要至少 `2 * seasonality` 个数据点
4. 无法转换的行会被自动丢弃，并在 `dropped` 字段中报告
5. 如果文件不存在或格式不支持，会返回错误消息，不会抛出未捕获的异常
6. **推荐使用智能检测工具**：使用 `detect_with_auto_convert` 工具，它会自动处理转换和检测，确保每次检测前文件都已正确转换
7. **避免"已转换"误报**：`auto_convert_data` 在转换指定文件时会检查输出文件是否已存在，避免重复转换
8. **文件路径**：支持相对路径（如 `models/file.csv`）和绝对路径，相对路径相对于项目根目录
9. **输出格式**：转换后的文件为CSV格式，包含 `time` 和 `value` 两列
10. data文件夹中存放原始数据，models文件夹里面存放的是经过统一格式转化的数据，询问AI的时候prompt要求检测转化格式后的数据。

## 技术栈

- Python 3.11.13
- fastmcp: MCP 框架
- numpy: 数值计算
- pandas: 数据处理
- statsmodels: STL 分解
- scipy: 统计函数

## 代码结构说明

- **mcp_server.py**: MCP 服务器封装，负责工具注册和服务器启动
- **sh_esd_tool.py**: 工具实现模块，包含：
  - 数据加载和转换 (`load_and_convert_data`) - 支持CSV、TXT、JSON格式
  - 季节性周期推断 (`infer_seasonality`)
  - STL 分解 (`stl_decompose`)
  - GESD 异常检测 (`gesd_test`)
  - 主检测函数 (`detect_sh_esd_anomaly`)
- **data_converter.py**: 数据转换工具，包含：
  - 日志文件转换 (`convert_log_to_timeseries`) - 支持HDFS、OpenSSH、Spark等日志格式
  - CSV/TXT转换 (`convert_csv_to_timeseries`) - 支持标准时间序列数据
  - 命令行工具接口

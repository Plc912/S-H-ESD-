#!/usr/bin/env python3
"""
S-H-ESD MCP 服务器封装
使用 fastmcp 框架，SSE 传输，监听 127.0.0.1:2701
"""

import os
from pathlib import Path
from fastmcp import FastMCP

# 创建 MCP 服务器实例
mcp = FastMCP("S-H-ESD Anomaly Detection Server")

# 获取项目根目录（脚本所在目录）
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def resolve_path(file_path: str) -> Path:
    """
    解析文件路径，支持相对路径和绝对路径
    如果相对路径，则相对于项目根目录
    """
    path = Path(file_path)
    if path.is_absolute():
        return path
    # 相对路径，尝试相对于项目根目录
    return PROJECT_ROOT / path


# 注册工具
@mcp.tool()
def detect_sh_esd_anomaly(
    file_path: str,
    seasonality: int = None,
    alpha: float = 0.05,
    max_anomalies: int = None,
    format_opts: dict = None
):
    """
    使用季节性混合 ESD (S-H-ESD) 方法检测时间序列异常
    
    Args:
        file_path: 数据文件路径（支持相对路径如'models/file.csv'或绝对路径，CSV、TXT或JSON格式，UTF-8编码）
        seasonality: 季节性周期长度（可选，默认自动推断）
        alpha: 显著性水平（默认 0.05）
        max_anomalies: 最大检测异常数（可选，默认自动）
        format_opts: 格式选项字典，可包含：
            - time_col: 时间列名（默认 'time'）
            - value_col: 数值列名（默认 'value'）
            - delimiter: 分隔符（CSV 默认 ','，TXT 默认 '\\t'）
    
    Returns:
        包含异常检测结果的字典
    """
    from sh_esd_tool import detect_sh_esd_anomaly as _detect_func
    # 解析文件路径
    resolved_path = resolve_path(file_path)
    return _detect_func(
        file_path=str(resolved_path),
        seasonality=seasonality,
        alpha=alpha,
        max_anomalies=max_anomalies,
        format_opts=format_opts
    )


# 注册数据转换工具
@mcp.tool()
def convert_data_to_timeseries(
    input_file: str,
    output_file: str = None,
    time_col: str = "time",
    value_col: str = "value",
    delimiter: str = ",",
    aggregation_window: str = "1min",
    log_mode: bool = False
):
    """
    将不同格式的数据文件转换为统一的时间序列格式（CSV格式）
    转换后的数据保存在models文件夹中，包含'time'和'value'两列
    
    Args:
        input_file: 输入文件路径（CSV、TXT或日志文件）
        output_file: 输出文件路径（可选，默认：models/输入文件名.json）
        time_col: 时间列名（仅用于标准CSV/TXT，默认：'time'）
        value_col: 数值列名（仅用于标准CSV/TXT，默认：'value'）
        delimiter: 分隔符（仅用于CSV/TXT，默认：','）
        aggregation_window: 聚合窗口（仅用于日志文件，如'1min', '5min', '1H'，默认：'1min'）
        log_mode: 是否强制使用日志模式（按时间聚合日志数量，默认：False，自动检测）
    
    Returns:
        包含转换结果的字典
    """
    from data_converter import convert_log_to_timeseries, convert_csv_to_timeseries
    from pathlib import Path
    import os
    
    try:
        # 解析输入文件路径
        input_path = resolve_path(input_file)
        if not input_path.exists():
            return {
                "success": False,
                "input_file": str(input_path),
                "error": f"文件不存在: {input_path}",
                "message": f"错误: 文件不存在 - {input_path}"
            }
        
        # 确定输出路径
        if output_file is None:
            input_name = input_path.stem
            output_path = MODELS_DIR / f"{input_name}.csv"
        else:
            output_path = resolve_path(output_file)
            # 确保扩展名是.csv
            if output_path.suffix.lower() != '.csv':
                output_path = output_path.with_suffix('.csv')
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 判断文件类型
        file_name = input_path.stem.lower()
        is_log_file = log_mode or any(keyword in file_name for keyword in ['hdfs', 'openssh', 'spark', 'log'])
        
        if is_log_file:
            result = convert_log_to_timeseries(
                str(input_path),
                aggregation_window=aggregation_window,
                output_path=str(output_path)
            )
        else:
            result = convert_csv_to_timeseries(
                str(input_path),
                time_col=time_col,
                value_col=value_col,
                output_path=str(output_path),
                delimiter=delimiter
            )
        
        return {
            "success": True,
            "input_file": str(input_path),
            "output_file": str(output_path),
            "records": len(result),
            "time_range": {
                "start": result[0]['time'] if result else None,
                "end": result[-1]['time'] if result else None
            },
            "value_range": {
                "min": min(r['value'] for r in result) if result else None,
                "max": max(r['value'] for r in result) if result else None
            },
            "message": f"成功转换 {len(result)} 条记录到 {output_path}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "input_file": input_file,
            "output_file": str(output_path) if 'output_path' in locals() else None,
            "error": str(e),
            "message": f"转换失败: {str(e)}"
        }


# 注册列出数据文件工具
@mcp.tool()
def list_data_files():
    """
    列出data文件夹中的所有数据文件
    
    Returns:
        包含文件列表的字典
    """
    try:
        # 确保data目录存在
        DATA_DIR.mkdir(exist_ok=True)
        
        # 列出所有CSV和TXT文件
        files = []
        for ext in ['.csv', '.txt']:
            for file_path in DATA_DIR.glob(f'*{ext}'):
                file_info = {
                    "name": file_path.name,
                    "path": str(file_path),
                    "relative_path": f"data/{file_path.name}",
                    "size": file_path.stat().st_size,
                    "exists": True
                }
                files.append(file_info)
        
        # 检查models目录中是否已有转换后的文件
        MODELS_DIR.mkdir(exist_ok=True)
        converted_files = {f.stem for f in MODELS_DIR.glob('*.csv')}
        
        # 标记哪些文件已转换
        for file_info in files:
            file_info["converted"] = file_info["name"].replace('.csv', '').replace('.txt', '') in converted_files
        
        return {
            "success": True,
            "data_dir": str(DATA_DIR),
            "models_dir": str(MODELS_DIR),
            "files": files,
            "count": len(files),
            "message": f"找到 {len(files)} 个数据文件"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"列出文件失败: {str(e)}"
        }


# 注册自动转换工具
@mcp.tool()
def auto_convert_data(
    file_name: str = None,
    aggregation_window: str = "1min",
    log_mode: bool = False
):
    """
    自动转换data文件夹中的数据文件到models文件夹（CSV格式）
    如果不指定file_name，则转换所有未转换的文件
    
    Args:
        file_name: 要转换的文件名（可选，默认转换所有文件）
        aggregation_window: 聚合窗口（仅用于日志文件，默认：'1min'）
        log_mode: 是否强制使用日志模式（默认：False，自动检测）
    
    Returns:
        包含转换结果的字典
    """
    try:
        # 确保目录存在
        DATA_DIR.mkdir(exist_ok=True)
        MODELS_DIR.mkdir(exist_ok=True)
        
        # 获取要转换的文件列表
        if file_name:
            # 转换指定文件
            input_path = DATA_DIR / file_name
            if not input_path.exists():
                return {
                    "success": False,
                    "error": f"文件不存在: {input_path}",
                    "message": f"错误: 文件不存在 - {input_path}"
                }
            # 检查输出文件是否已存在
            output_path = MODELS_DIR / f"{input_path.stem}.csv"
            if output_path.exists():
                return {
                    "success": True,
                    "converted": [],
                    "skipped": [{
                        "input_file": input_path.name,
                        "output_file": output_path.name,
                        "reason": "文件已存在，跳过转换"
                    }],
                    "message": f"文件 {input_path.name} 已转换，输出文件 {output_path.name} 已存在"
                }
            files_to_convert = [input_path]
        else:
            # 转换所有未转换的文件
            converted_stems = {f.stem for f in MODELS_DIR.glob('*.csv')}
            files_to_convert = [
                f for f in DATA_DIR.glob('*.csv') + list(DATA_DIR.glob('*.txt'))
                if f.stem not in converted_stems
            ]
        
        if not files_to_convert:
            return {
                "success": True,
                "converted": [],
                "skipped": [],
                "message": "没有需要转换的文件（所有文件已转换）"
            }
        
        # 转换文件
        from data_converter import convert_log_to_timeseries, convert_csv_to_timeseries
        
        converted = []
        failed = []
        
        for input_path in files_to_convert:
            try:
                output_path = MODELS_DIR / f"{input_path.stem}.csv"
                
                # 判断文件类型
                file_name_lower = input_path.stem.lower()
                is_log_file = log_mode or any(keyword in file_name_lower for keyword in ['hdfs', 'openssh', 'spark', 'log'])
                
                if is_log_file:
                    result = convert_log_to_timeseries(
                        str(input_path),
                        aggregation_window=aggregation_window,
                        output_path=str(output_path)
                    )
                else:
                    result = convert_csv_to_timeseries(
                        str(input_path),
                        output_path=str(output_path)
                    )
                
                converted.append({
                    "input_file": input_path.name,
                    "output_file": output_path.name,
                    "records": len(result)
                })
            except Exception as e:
                failed.append({
                    "input_file": input_path.name,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "converted": converted,
            "failed": failed,
            "total": len(files_to_convert),
            "success_count": len(converted),
            "failed_count": len(failed),
            "message": f"成功转换 {len(converted)} 个文件，失败 {len(failed)} 个文件"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"自动转换失败: {str(e)}"
        }


# 注册智能检测工具（自动转换+检测）
@mcp.tool()
def detect_with_auto_convert(
    data_file: str,
    seasonality: int = None,
    alpha: float = 0.05,
    max_anomalies: int = None,
    aggregation_window: str = "1min",
    log_mode: bool = False
):
    """
    智能异常检测工具：自动转换数据文件（如果需要）然后进行异常检测
    如果data文件夹中的文件尚未转换，会自动转换到models文件夹，然后进行检测
    
    Args:
        data_file: data文件夹中的文件路径（如 'data/HDFS_2k.log_structured.csv' 或 'HDFS_2k.log_structured.csv'）
        seasonality: 季节性周期长度（可选，默认自动推断）
        alpha: 显著性水平（默认 0.05）
        max_anomalies: 最大检测异常数（可选，默认自动）
        aggregation_window: 聚合窗口（仅用于日志文件，默认：'1min'）
        log_mode: 是否强制使用日志模式（默认：False，自动检测）
    
    Returns:
        包含转换和检测结果的字典
    """
    try:
        # 解析输入文件路径
        if data_file.startswith('data/'):
            file_name = data_file.replace('data/', '')
        elif data_file.startswith('@data/'):
            file_name = data_file.replace('@data/', '')
        else:
            file_name = data_file
        
        input_path = DATA_DIR / file_name
        
        if not input_path.exists():
            return {
                "success": False,
                "error": f"文件不存在: {input_path}",
                "message": f"错误: data文件夹中不存在文件 {file_name}",
                "detection_result": None
            }
        
        # 确定输出文件路径
        output_path = MODELS_DIR / f"{input_path.stem}.csv"
        
        # 检查是否需要转换
        need_convert = not output_path.exists()
        conversion_result = None
        
        if need_convert:
            # 自动转换文件
            from data_converter import convert_log_to_timeseries, convert_csv_to_timeseries
            
            # 判断文件类型
            file_name_lower = input_path.stem.lower()
            is_log_file = log_mode or any(keyword in file_name_lower for keyword in ['hdfs', 'openssh', 'spark', 'log'])
            
            try:
                if is_log_file:
                    result = convert_log_to_timeseries(
                        str(input_path),
                        aggregation_window=aggregation_window,
                        output_path=str(output_path)
                    )
                else:
                    result = convert_csv_to_timeseries(
                        str(input_path),
                        output_path=str(output_path)
                    )
                
                conversion_result = {
                    "success": True,
                    "converted": True,
                    "input_file": input_path.name,
                    "output_file": output_path.name,
                    "records": len(result),
                    "message": f"成功转换 {len(result)} 条记录"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"转换失败: {str(e)}",
                    "message": f"转换文件 {file_name} 失败: {str(e)}",
                    "conversion_result": None,
                    "detection_result": None
                }
        else:
            conversion_result = {
                "success": True,
                "converted": False,
                "input_file": input_path.name,
                "output_file": output_path.name,
                "message": "文件已存在，跳过转换"
            }
        
        # 进行异常检测
        from sh_esd_tool import detect_sh_esd_anomaly as _detect_func
        
        try:
            detection_result = _detect_func(
                file_path=str(output_path),
                seasonality=seasonality,
                alpha=alpha,
                max_anomalies=max_anomalies,
                format_opts=None
            )
            
            # 检查检测结果
            if detection_result.get('n', 0) < 3:
                return {
                    "success": False,
                    "data_file": str(input_path),
                    "conversion_result": conversion_result,
                    "detection_result": detection_result,
                    "error": "数据点不足",
                    "message": f"检测失败: {detection_result.get('message', '数据点不足，至少需要3个数据点才能进行异常检测')}"
                }
            
            return {
                "success": True,
                "data_file": str(input_path),
                "conversion_result": conversion_result,
                "detection_result": detection_result,
                "message": f"检测完成: {detection_result.get('message', 'N/A')}"
            }
        except Exception as e:
            return {
                "success": False,
                "data_file": str(input_path),
                "conversion_result": conversion_result,
                "error": f"检测失败: {str(e)}",
                "message": f"异常检测失败: {str(e)}。请检查输入文件格式或参数设置是否正确。可能需要重新尝试转换或调整检测参数。",
                "detection_result": None
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"智能检测失败: {str(e)}",
            "conversion_result": None,
            "detection_result": None
        }


def main():
    """启动 MCP 服务器"""
    host = "127.0.0.1"
    port = 2701
    
    print(f"启动 S-H-ESD MCP 服务器...")
    print(f"监听地址: {host}:{port}")
    print(f"传输方式: SSE")
    print(f"工具:")
    print(f"  - detect_with_auto_convert: 智能检测（推荐）- 自动转换并检测")
    print(f"  - detect_sh_esd_anomaly: 异常检测（需先转换）")
    print(f"  - list_data_files: 列出data文件夹中的文件")
    print(f"  - auto_convert_data: 自动转换data文件夹中的文件")
    print(f"  - convert_data_to_timeseries: 手动转换指定文件")
    print(f"  - auto_convert_data: 自动转换data文件夹中的文件")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"模型目录: {MODELS_DIR}")
    print(f"SSE 端点: http://{host}:{port}/sse")
    
    # 使用 fastmcp 的 run 方法，指定 SSE 传输
    mcp.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    main()

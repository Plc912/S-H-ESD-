#!/usr/bin/env python3
"""
数据转换工具
将不同格式的数据文件转换为统一的时间序列格式（list[dict]，每个dict包含'time'和'value'键）
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def parse_hdfs_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """解析HDFS格式的日期时间：081109, 203615 -> 2008-11-09 20:36:15"""
    try:
        # 日期格式：YYMMDD (081109 = 2008-11-09)
        # 转换为字符串并补齐前导零
        date_str = str(date_str).zfill(6)
        year_part = int(date_str[:2])
        # HDFS日志通常是2008年的数据
        # 如果年份小于50，认为是20xx年，否则是19xx年
        if year_part < 50:
            year = 2000 + year_part
        else:
            year = 1900 + year_part
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        
        # 时间格式：HHMMSS (203615 = 20:36:15)
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        
        return datetime(year, month, day, hour, minute, second)
    except Exception:
        return None


def parse_openssh_datetime(date_str: str, day_str: str, time_str: str) -> Optional[datetime]:
    """解析OpenSSH格式的日期时间：Dec, 10, 06:55:46"""
    try:
        # 月份映射
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        
        month = month_map.get(date_str, 1)
        day = int(day_str)
        year = 2008  # 默认年份，因为数据中没有年份信息
        
        # 时间格式：HH:MM:SS
        time_parts = time_str.split(':')
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = int(time_parts[2])
        
        return datetime(year, month, day, hour, minute, second)
    except Exception:
        return None


def parse_spark_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """解析Spark格式的日期时间：17/06/09, 20:10:40 -> 2017-06-09 20:10:40"""
    try:
        # 日期格式：YY/MM/DD (17/06/09 = 2017-06-09)
        date_parts = date_str.split('/')
        year = 2000 + int(date_parts[0])
        month = int(date_parts[1])
        day = int(date_parts[2])
        
        # 时间格式：HH:MM:SS
        time_parts = time_str.split(':')
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = int(time_parts[2])
        
        return datetime(year, month, day, hour, minute, second)
    except Exception:
        return None


def convert_log_to_timeseries(
    file_path: str,
    aggregation_window: str = '1min',
    output_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    将日志文件转换为时间序列数据
    
    Args:
        file_path: 输入文件路径
        aggregation_window: 聚合窗口（'1min', '5min', '1H'等）
        output_path: 输出文件路径（可选，如果提供则保存为JSON）
    
    Returns:
        时间序列数据列表，每个元素包含'time'和'value'键
    """
    file_ext = Path(file_path).suffix.lower()
    file_name = Path(file_path).stem
    
    # 读取CSV文件
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        raise ValueError(f"读取文件失败: {str(e)}")
    
    if df.empty:
        raise ValueError("文件为空")
    
    # 根据文件类型解析时间
    timestamps = []
    
    # 优先检查文件名，然后检查列组合
    # Spark格式：有Level列，Date格式为YY/MM/DD
    if 'Spark' in file_name or ('Date' in df.columns and 'Time' in df.columns and 'Level' in df.columns):
        # Spark格式
        for idx, row in df.iterrows():
            dt = parse_spark_datetime(str(row['Date']), str(row['Time']))
            if dt:
                timestamps.append(dt)
    
    # OpenSSH格式：有Day列
    elif 'OpenSSH' in file_name or ('Date' in df.columns and 'Day' in df.columns and 'Time' in df.columns):
        # OpenSSH格式
        for idx, row in df.iterrows():
            dt = parse_openssh_datetime(str(row['Date']), str(row['Day']), str(row['Time']))
            if dt:
                timestamps.append(dt)
    
    # HDFS格式：有Date和Time列，但没有Day和Level列
    elif 'HDFS' in file_name or ('Date' in df.columns and 'Time' in df.columns and 'Day' not in df.columns and 'Level' not in df.columns):
        # HDFS格式
        for idx, row in df.iterrows():
            dt = parse_hdfs_datetime(str(row['Date']), str(row['Time']))
            if dt:
                timestamps.append(dt)
    
    else:
        # 尝试自动检测时间列
        time_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['time', 'date', 'timestamp'])]
        if not time_cols:
            raise ValueError(f"无法识别文件格式。可用列: {list(df.columns)}")
        
        # 尝试使用pandas自动解析
        time_col = time_cols[0]
        for idx, row in df.iterrows():
            try:
                dt = pd.to_datetime(row[time_col], errors='coerce')
                if pd.notna(dt):
                    timestamps.append(dt.to_pydatetime())
            except:
                continue
    
    if not timestamps:
        raise ValueError("未能解析任何有效的时间戳")
    
    # 按时间聚合（统计每个时间窗口的日志数量）
    df_timeseries = pd.DataFrame({'timestamp': timestamps})
    df_timeseries['timestamp'] = pd.to_datetime(df_timeseries['timestamp'])
    df_timeseries = df_timeseries.set_index('timestamp')
    
    # 自动检测时间跨度，选择合适的聚合窗口
    time_span = (df_timeseries.index.max() - df_timeseries.index.min()).total_seconds()
    n_records = len(df_timeseries)
    
    # 如果时间跨度很短，自动调整聚合窗口
    if time_span < 300:  # 小于5分钟
        if time_span < 60:  # 小于1分钟，使用1秒窗口
            auto_window = '1s'
        elif time_span < 300:  # 小于5分钟，使用5秒窗口
            auto_window = '5s'
        else:
            auto_window = aggregation_window
    else:
        auto_window = aggregation_window
    
    # 确保聚合后至少有10个数据点
    try:
        estimated_points = time_span / pd.Timedelta(auto_window).total_seconds()
        if estimated_points < 10 and time_span > 0:
            # 调整窗口大小，确保至少有10个点
            target_window_seconds = time_span / 10
            if target_window_seconds < 1:
                auto_window = '1s'
            elif target_window_seconds < 5:
                auto_window = '5s'
            elif target_window_seconds < 10:
                auto_window = '10s'
            else:
                auto_window = aggregation_window
    except:
        # 如果无法解析窗口，使用默认值
        pass
    
    # 按时间窗口聚合
    aggregated = df_timeseries.resample(auto_window).size()
    
    # 转换为统一格式
    result = []
    for timestamp, count in aggregated.items():
        result.append({
            'time': timestamp.isoformat(),
            'value': float(count)
        })
    
    # 按时间排序
    result.sort(key=lambda x: x['time'])
    
    # 如果指定了输出路径，保存为CSV
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        # 转换为DataFrame并保存为CSV
        df_output = pd.DataFrame(result)
        df_output.to_csv(output_path, index=False, encoding='utf-8')
        print(f"转换完成，已保存到: {output_path}")
        print(f"共生成 {len(result)} 个数据点")
    
    return result


def convert_csv_to_timeseries(
    file_path: str,
    time_col: str = 'time',
    value_col: str = 'value',
    output_path: Optional[str] = None,
    delimiter: str = ','
) -> List[Dict[str, Any]]:
    """
    将标准CSV文件转换为时间序列数据
    
    Args:
        file_path: 输入文件路径
        time_col: 时间列名
        value_col: 数值列名
        output_path: 输出文件路径（可选）
        delimiter: 分隔符
    
    Returns:
        时间序列数据列表
    """
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
    except Exception as e:
        raise ValueError(f"读取文件失败: {str(e)}")
    
    if df.empty:
        raise ValueError("文件为空")
    
    if time_col not in df.columns:
        raise ValueError(f"未找到时间列 '{time_col}'。可用列: {list(df.columns)}")
    
    if value_col not in df.columns:
        raise ValueError(f"未找到数值列 '{value_col}'。可用列: {list(df.columns)}")
    
    result = []
    dropped = 0
    
    for idx, row in df.iterrows():
        try:
            # 解析时间
            time_val = row[time_col]
            if pd.isna(time_val):
                dropped += 1
                continue
            
            # 转换为datetime
            if isinstance(time_val, (int, float)) and time_val > 1e10:
                time_val = pd.to_datetime(time_val, unit='ms')
            elif isinstance(time_val, (int, float)) and time_val > 1e9:
                time_val = pd.to_datetime(time_val, unit='s')
            else:
                time_val = pd.to_datetime(time_val, errors='coerce')
            
            if pd.isna(time_val):
                dropped += 1
                continue
            
            # 解析数值
            value_val = row[value_col]
            if pd.isna(value_val):
                dropped += 1
                continue
            
            result.append({
                'time': time_val.isoformat(),
                'value': float(value_val)
            })
        except Exception:
            dropped += 1
            continue
    
    if not result:
        raise ValueError("没有有效的数据点")
    
    # 按时间排序
    result.sort(key=lambda x: x['time'])
    
    if dropped > 0:
        print(f"警告: 丢弃了 {dropped} 行无效数据")
    
    # 如果指定了输出路径，保存为CSV
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        # 转换为DataFrame并保存为CSV
        df_output = pd.DataFrame(result)
        df_output.to_csv(output_path, index=False, encoding='utf-8')
        print(f"转换完成，已保存到: {output_path}")
        print(f"共生成 {len(result)} 个数据点")
    
    return result


def main():
    """命令行工具入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据转换工具：将不同格式的数据转换为统一的时间序列格式')
    parser.add_argument('input_file', help='输入文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径（默认：models/输入文件名.json）')
    parser.add_argument('--time-col', default='time', help='时间列名（默认：time）')
    parser.add_argument('--value-col', default='value', help='数值列名（默认：value）')
    parser.add_argument('--delimiter', default=',', help='分隔符（默认：,）')
    parser.add_argument('--aggregation', default='1min', help='聚合窗口（仅用于日志文件，默认：1min）')
    parser.add_argument('--log-mode', action='store_true', help='强制使用日志模式（按时间聚合）')
    
    args = parser.parse_args()
    
    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        input_name = Path(args.input_file).stem
        output_path = f"models/{input_name}.json"
    
    # 判断文件类型
    file_name = Path(args.input_file).stem.lower()
    is_log_file = args.log_mode or any(keyword in file_name for keyword in ['hdfs', 'openssh', 'spark', 'log'])
    
    try:
        if is_log_file:
            result = convert_log_to_timeseries(
                args.input_file,
                aggregation_window=args.aggregation,
                output_path=output_path
            )
        else:
            result = convert_csv_to_timeseries(
                args.input_file,
                time_col=args.time_col,
                value_col=args.value_col,
                output_path=output_path,
                delimiter=args.delimiter
            )
        
        print(f"成功转换 {len(result)} 条记录")
        if result:
            print(f"时间范围: {result[0]['time']} 到 {result[-1]['time']}")
            print(f"数值范围: {min(r['value'] for r in result):.2f} 到 {max(r['value'] for r in result):.2f}")
    
    except Exception as e:
        print(f"错误: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

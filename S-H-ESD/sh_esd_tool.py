#!/usr/bin/env python3
"""
S-H-ESD (Seasonal Hybrid ESD) 时间序列异常检测工具实现
包含数据加载、STL分解、GESD检测等核心功能
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()


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


def infer_seasonality(time_series: List[Dict[str, Any]], n: int) -> int:
    """
    自动推断季节性周期长度
    
    Args:
        time_series: 时间序列数据
        n: 数据点总数
    
    Returns:
        推断的季节性周期长度
    """
    if n < 20:
        return max(2, n // 4)
    
    # 尝试推断常见的周期
    # 如果数据是日级别的，可能周期为 7（周）、30（月）等
    # 如果数据是小时级别的，可能周期为 24（日）、168（周）等
    
    # 简单策略：使用 n//4 作为默认值，但至少为 2
    # 更复杂的策略可以基于自相关函数
    seasonality = max(2, n // 4)
    
    # 限制最大周期为 n//2
    seasonality = min(seasonality, n // 2)
    
    return seasonality


def load_and_convert_data(
    file_path: str,
    format_opts: Optional[Dict[str, Any]] = None
) -> tuple[List[Dict[str, Any]], int]:
    """
    加载并转换数据文件为统一格式
    
    Args:
        file_path: 文件路径（支持CSV、TXT、JSON格式）
        format_opts: 格式选项（列名映射、分隔符等）
    
    Returns:
        (转换后的数据列表, 丢弃的行数)
    """
    if format_opts is None:
        format_opts = {}
    
    # 确定文件类型
    file_ext = Path(file_path).suffix.lower()
    is_csv = file_ext == '.csv'
    is_txt = file_ext == '.txt'
    is_json = file_ext == '.json'
    
    # 如果是JSON文件，直接加载
    if is_json:
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据格式
            if not isinstance(data, list):
                raise ValueError("JSON文件必须包含一个数组")
            
            # 验证每个元素是否包含time和value
            result = []
            dropped = 0
            for item in data:
                if isinstance(item, dict) and 'time' in item and 'value' in item:
                    try:
                        # 确保value是数值
                        item['value'] = float(item['value'])
                        result.append({
                            'time': str(item['time']),
                            'value': item['value']
                        })
                    except (ValueError, TypeError):
                        dropped += 1
                else:
                    dropped += 1
            
            if not result:
                raise ValueError("JSON文件中没有有效的数据点")
            
            # 按时间排序
            result.sort(key=lambda x: x['time'])
            return result, dropped
        
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析失败: {str(e)}")
        except Exception as e:
            raise ValueError(f"读取JSON文件失败: {str(e)}")
    
    if not (is_csv or is_txt):
        raise ValueError(f"不支持的文件类型: {file_ext}。仅支持 .csv、.txt 和 .json 文件")
    
    # 确定分隔符
    delimiter = format_opts.get('delimiter', ',' if is_csv else '\t')
    
    # 读取文件
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
    except Exception as e:
        raise ValueError(f"读取文件失败: {str(e)}")
    
    if df.empty:
        raise ValueError("文件为空")
    
    # 确定列名映射
    time_col = format_opts.get('time_col', 'time')
    value_col = format_opts.get('value_col', 'value')
    
    # 如果指定的列不存在，尝试查找类似的列
    if time_col not in df.columns:
        # 尝试查找包含 'time', 'date', 'timestamp' 的列
        time_candidates = [col for col in df.columns 
                          if any(keyword in col.lower() 
                                for keyword in ['time', 'date', 'timestamp'])]
        if time_candidates:
            time_col = time_candidates[0]
        else:
            raise ValueError(f"未找到时间列。可用列: {list(df.columns)}")
    
    if value_col not in df.columns:
        # 尝试查找数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            value_col = numeric_cols[0]
        else:
            raise ValueError(f"未找到数值列。可用列: {list(df.columns)}")
    
    # 转换数据
    result = []
    dropped = 0
    
    for idx, row in df.iterrows():
        try:
            # 解析时间
            time_val = row[time_col]
            if pd.isna(time_val):
                dropped += 1
                continue
            
            # 转换为 datetime
            if isinstance(time_val, (int, float)) and time_val > 1e10:
                # 可能是 UNIX 时间戳（毫秒）
                time_val = pd.to_datetime(time_val, unit='ms')
            elif isinstance(time_val, (int, float)) and time_val > 1e9:
                # 可能是 UNIX 时间戳（秒）
                time_val = pd.to_datetime(time_val, unit='s')
            else:
                time_val = pd.to_datetime(time_val, errors='coerce')
            
            if pd.isna(time_val):
                dropped += 1
                continue
            
            # 转换为 ISO8601 字符串
            time_str = time_val.isoformat()
            
            # 解析数值
            value_val = row[value_col]
            if pd.isna(value_val):
                dropped += 1
                continue
            
            value_float = float(value_val)
            
            result.append({
                'time': time_str,
                'value': value_float
            })
        except Exception:
            dropped += 1
            continue
    
    if not result:
        raise ValueError("没有有效的数据点")
    
    # 按时间排序
    result.sort(key=lambda x: x['time'])
    
    return result, dropped


def stl_decompose(values: np.ndarray, seasonality: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 STL 分解时间序列
    
    Args:
        values: 数值数组
        seasonality: 季节性周期
    
    Returns:
        (seasonal, trend, residual)
    """
    n = len(values)
    
    # 如果数据点太少，使用简单分解
    if n < seasonality * 2:
        # 简单移动平均作为趋势
        window = min(seasonality, n // 2)
        if window < 2:
            trend = np.full(n, np.mean(values))
        else:
            trend_series = pd.Series(values).rolling(window=window, center=True).mean()
            trend = trend_series.bfill().ffill().values
        
        # 残差 = 原始值 - 趋势
        residual = values - trend
        seasonal = np.zeros(n)
        return seasonal, trend, residual
    
    # 使用 statsmodels STL 分解
    try:
        # 确保 seasonality 是奇数（STL 要求）
        period = seasonality if seasonality % 2 == 1 else seasonality + 1
        
        # 限制 period 不超过数据长度的一半
        period = min(period, n // 2)
        if period < 2:
            period = 2
        
        stl = STL(pd.Series(values), seasonal=period, robust=True)
        result = stl.fit()
        
        seasonal = result.seasonal.values
        trend = result.trend.values
        residual = result.resid.values
        
        return seasonal, trend, residual
    except Exception as e:
        # 如果 STL 失败，使用简单分解
        window = min(seasonality, n // 2)
        if window < 2:
            trend = np.full(n, np.mean(values))
        else:
            trend_series = pd.Series(values).rolling(window=window, center=True).mean()
            trend = trend_series.bfill().ffill().values
        
        residual = values - trend
        seasonal = np.zeros(n)
        return seasonal, trend, residual


def gesd_test(residuals: np.ndarray, alpha: float = 0.05, max_anomalies: Optional[int] = None) -> tuple[List[int], List[float], List[float]]:
    """
    执行 Generalized ESD (Extreme Studentized Deviate) 测试
    
    Args:
        residuals: 残差数组
        alpha: 显著性水平
        max_anomalies: 最大异常数（None 表示自动）
    
    Returns:
        (异常索引列表, 异常分数列表, z-score 列表)
    """
    n = len(residuals)
    if n < 3:
        return [], [], []
    
    # 计算所有点的 z-score
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    
    if std_res == 0:
        return [], [], []
    
    zscores = np.abs((residuals - mean_res) / std_res).tolist()
    
    # 确定最大异常数
    if max_anomalies is None:
        # 自动计算：最多检测 n//10 个异常，但至少需要 3 个数据点
        max_anomalies = max(1, min(n // 10, n - 2))
    
    max_anomalies = min(max_anomalies, n - 2)
    
    # GESD 迭代过程
    anomaly_indices = []
    working_residuals = residuals.copy()
    working_indices = np.arange(n)
    
    for i in range(max_anomalies):
        if len(working_residuals) < 2:
            break
        
        # 计算当前残差的统计量
        mean_w = np.mean(working_residuals)
        std_w = np.std(working_residuals)
        
        if std_w == 0:
            break
        
        # 找到最极端的点
        abs_deviations = np.abs(working_residuals - mean_w)
        max_idx = np.argmax(abs_deviations)
        R_i = abs_deviations[max_idx] / std_w
        
        # 计算临界值 lambda_i
        # 使用 t 分布
        p = 1 - alpha / (2 * (n - i))
        df = len(working_residuals) - 2
        if df < 1:
            break
        
        try:
            lambda_i = stats.t.ppf(p, df) * (n - i - 1) / np.sqrt((n - i - 2 + stats.t.ppf(p, df)**2) * (n - i))
        except:
            lambda_i = 3.0  # 默认阈值
        
        # 判断是否为异常
        if R_i > lambda_i:
            # 记录异常点
            original_idx = int(working_indices[max_idx])
            anomaly_indices.append(original_idx)
            
            # 移除异常点
            working_residuals = np.delete(working_residuals, max_idx)
            working_indices = np.delete(working_indices, max_idx)
        else:
            # 不再有异常点
            break
    
    # 计算异常分数（基于 z-score，归一化到 [0,1]）
    scores = []
    if anomaly_indices:
        max_zscore = max(zscores[i] for i in anomaly_indices) if anomaly_indices else 1.0
        for idx in range(n):
            if idx in anomaly_indices:
                # 异常点的分数基于其 z-score
                score = min(1.0, 0.5 + 0.5 * (zscores[idx] / max(3.0, max_zscore)))
            else:
                # 非异常点的分数较低
                score = min(0.5, zscores[idx] / 6.0)
            scores.append(score)
    else:
        scores = [min(0.5, z / 6.0) for z in zscores]
    
    return sorted(anomaly_indices), scores, zscores


def detect_sh_esd_anomaly(
    file_path: str,
    seasonality: Optional[int] = None,
    alpha: float = 0.05,
    max_anomalies: Optional[int] = None,
    format_opts: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    使用季节性混合 ESD (S-H-ESD) 方法检测时间序列异常
    
    Args:
        file_path: 数据文件路径（CSV、TXT 或 JSON 格式，UTF-8 编码）
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
    try:
        # 解析文件路径
        resolved_path = resolve_path(file_path)
        
        # 检查文件是否存在
        if not resolved_path.exists():
            return {
                "is_anomalous": False,
                "n": 0,
                "seasonality": 0,
                "alpha": alpha,
                "max_anomalies": max_anomalies,
                "k_found": 0,
                "iqr_lower": 0.0,
                "iqr_upper": 0.0,
                "anomaly_indices": [],
                "anomaly_times": [],
                "scores": [],
                "zscores": [],
                "dropped": 0,
                "message": f"错误: 文件不存在 - {resolved_path}"
            }
        
        # 加载和转换数据
        try:
            time_series, dropped = load_and_convert_data(str(resolved_path), format_opts)
        except Exception as e:
            return {
                "is_anomalous": False,
                "n": 0,
                "seasonality": 0,
                "alpha": alpha,
                "max_anomalies": max_anomalies,
                "k_found": 0,
                "iqr_lower": 0.0,
                "iqr_upper": 0.0,
                "anomaly_indices": [],
                "anomaly_times": [],
                "scores": [],
                "zscores": [],
                "dropped": 0,
                "message": f"错误: 数据加载失败 - {str(e)}"
            }
        
        n = len(time_series)
        
        # 检查数据量
        if n < 3:
            return {
                "is_anomalous": False,
                "n": n,
                "seasonality": 0,
                "alpha": alpha,
                "max_anomalies": max_anomalies,
                "k_found": 0,
                "iqr_lower": 0.0,
                "iqr_upper": 0.0,
                "anomaly_indices": [],
                "anomaly_times": [],
                "scores": [],
                "zscores": [],
                "dropped": dropped,
                "message": f"数据点不足（n={n}），至少需要 3 个数据点。丢弃了 {dropped} 行。"
            }
        
        # 提取数值数组
        values = np.array([item['value'] for item in time_series])
        times = [item['time'] for item in time_series]
        
        # 推断季节性周期
        if seasonality is None:
            seasonality = infer_seasonality(time_series, n)
        
        # 检查数据量是否足够进行季节性分解
        if n < seasonality * 2:
            # 数据不足，使用简单方法
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                return {
                    "is_anomalous": False,
                    "n": n,
                    "seasonality": seasonality,
                    "alpha": alpha,
                    "max_anomalies": max_anomalies,
                    "k_found": 0,
                    "iqr_lower": float(mean_val),
                    "iqr_upper": float(mean_val),
                    "anomaly_indices": [],
                    "anomaly_times": [],
                    "scores": [0.0] * n,
                    "zscores": [0.0] * n,
                    "dropped": dropped,
                    "message": f"数据点不足（n={n} < 2*seasonality={2*seasonality}），无法进行季节性分解。数据无变化（std=0）。丢弃了 {dropped} 行。"
                }
            
            # 使用简单 z-score 方法
            zscores = np.abs((values - mean_val) / std_val).tolist()
            threshold = 3.0
            anomaly_indices = [i for i, z in enumerate(zscores) if z > threshold]
            
            # 计算 IQR
            q1, q3 = np.percentile(values, [25, 75])
            iqr_lower = q1 - 1.5 * (q3 - q1)
            iqr_upper = q3 + 1.5 * (q3 - q1)
            
            # 计算分数
            max_zscore = max(zscores) if zscores else 1.0
            scores = [min(1.0, 0.5 + 0.5 * (z / max(3.0, max_zscore))) if z > threshold else min(0.5, z / 6.0) for z in zscores]
            
            anomaly_times = [times[i] for i in anomaly_indices]
            
            return {
                "is_anomalous": len(anomaly_indices) > 0,
                "n": n,
                "seasonality": seasonality,
                "alpha": alpha,
                "max_anomalies": max_anomalies,
                "k_found": len(anomaly_indices),
                "iqr_lower": float(iqr_lower),
                "iqr_upper": float(iqr_upper),
                "anomaly_indices": anomaly_indices,
                "anomaly_times": anomaly_times,
                "scores": scores,
                "zscores": zscores,
                "dropped": dropped,
                "message": f"数据点不足（n={n} < 2*seasonality={2*seasonality}），使用简单 z-score 方法。检测到 {len(anomaly_indices)} 个异常点。丢弃了 {dropped} 行。"
            }
        
        # STL 分解
        try:
            seasonal, trend, residual = stl_decompose(values, seasonality)
        except Exception as e:
            return {
                "is_anomalous": False,
                "n": n,
                "seasonality": seasonality,
                "alpha": alpha,
                "max_anomalies": max_anomalies,
                "k_found": 0,
                "iqr_lower": 0.0,
                "iqr_upper": 0.0,
                "anomaly_indices": [],
                "anomaly_times": [],
                "scores": [],
                "zscores": [],
                "dropped": dropped,
                "message": f"错误: STL 分解失败 - {str(e)}。丢弃了 {dropped} 行。"
            }
        
        # GESD 测试
        anomaly_indices, scores, zscores = gesd_test(residual, alpha, max_anomalies)
        
        # 计算 IQR（基于残差）
        q1, q3 = np.percentile(residual, [25, 75])
        iqr_lower = float(q1 - 1.5 * (q3 - q1))
        iqr_upper = float(q3 + 1.5 * (q3 - q1))
        
        # 获取异常点的时间
        anomaly_times = [times[i] for i in anomaly_indices]
        
        # 构建消息
        msg_parts = []
        if dropped > 0:
            msg_parts.append(f"丢弃了 {dropped} 行无效数据")
        msg_parts.append(f"检测到 {len(anomaly_indices)} 个异常点")
        if len(anomaly_indices) > 0:
            msg_parts.append(f"异常索引: {anomaly_indices[:10]}{'...' if len(anomaly_indices) > 10 else ''}")
        message = "。".join(msg_parts) if msg_parts else "检测完成，未发现异常"
        
        return {
            "is_anomalous": len(anomaly_indices) > 0,
            "n": n,
            "seasonality": seasonality,
            "alpha": alpha,
            "max_anomalies": max_anomalies if max_anomalies else "auto",
            "k_found": len(anomaly_indices),
            "iqr_lower": iqr_lower,
            "iqr_upper": iqr_upper,
            "anomaly_indices": anomaly_indices,
            "anomaly_times": anomaly_times,
            "scores": scores,
            "zscores": zscores,
            "dropped": dropped,
            "message": message
        }
    
    except Exception as e:
        return {
            "is_anomalous": False,
            "n": 0,
            "seasonality": 0,
            "alpha": alpha,
            "max_anomalies": max_anomalies,
            "k_found": 0,
            "iqr_lower": 0.0,
            "iqr_upper": 0.0,
            "anomaly_indices": [],
            "anomaly_times": [],
            "scores": [],
            "zscores": [],
            "dropped": 0,
            "message": f"未预期的错误: {str(e)}"
        }

"""IO 工具模块：读配置、建结果目录、存 json。

设计理由
--------
1. 配置与代码分离：所有超参数（数据路径、划分种子、CLAHE 参数、学习率……）写在
   ``configs/*.yaml`` 里，代码只读配置。迁移到 OCTA 时只需新增一个 yaml，而不改代码，
   这正是阶段二"代码可复用"要证明的事情。选 YAML 而不是 JSON，是因为 YAML 支持注释，
   可以把"为什么取这个值"直接写在配置旁边。
2. 结果目录按 ``results/<实验名>/<时间戳>/`` 组织：每次运行一个新目录，旧结果永不被覆盖，
   便于对比不同参数下的实验，也满足"报告里的数字必须来自 results/ 下实际文件"的规则。
3. json 保存指标：人和程序都能读；``ensure_ascii=False`` 让中文键名可读；
   ``_to_builtin`` 把 numpy 标量/数组转成 Python 原生类型，否则 json 会报
   "Object of type float32 is not JSON serializable"。
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# 仓库根目录：本文件位于 <repo>/utils/io.py，所以上两级就是仓库根
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path) -> Path:
    """相对路径一律按仓库根目录解析，避免"在不同目录下运行结果不同"的问题。"""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path) -> dict:
    """读取 YAML 配置文件，返回普通 dict。"""
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_run_dir(exp_name: str, results_root="results", timestamp: str | None = None) -> Path:
    """创建并返回 ``results/<exp_name>/<timestamp>/``。

    时间戳精确到秒（YYYYmmdd-HHMMSS），默认取当前时间；可显式传入便于测试。
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = resolve_path(results_root) / exp_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _to_builtin(obj):
    """递归地把 numpy 类型转成 json 可序列化的 Python 原生类型。"""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):  # np.float32、np.int64、np.bool_ 等标量
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(data, path) -> Path:
    """把 dict/list 保存为缩进 2 格的 UTF-8 json，返回写入路径。"""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(data), f, ensure_ascii=False, indent=2)
    return path


def load_json(path):
    """读取 json 文件。"""
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)

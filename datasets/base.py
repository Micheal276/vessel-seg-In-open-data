"""数据集基类：定义所有血管分割数据集（DRIVE、ROSE、OCTA-500……）共同遵守的"契约"。

设计理由
--------
1. 统一返回格式。下游的经典方法、U-Net 训练、指标计算只认一个字典::

       {
         "image": float32 数组，H×W×C（彩色）或 H×W（单通道），取值 [0, 1]
         "label": uint8 数组，H×W，取值 {0, 1}，1 = 血管
         "fov":   uint8 数组，H×W，取值 {0, 1}，1 = 视野（Field of View）内
         "name":  str，样本名（如 "21"），用于保存结果和追溯
       }

   只要新数据集返回同样的字典，model / train / eval 就一行都不用改——这正是阶段二
   "迁移到 OCTA 只新增 Dataset 适配类和 config"的前提。OCTA 没有 FOV 时，子类返回全 1 的 fov 即可。

2. 模板方法模式。基类负责"公共流程"（按 split 取样本名、__len__、__getitem__、格式校验），
   子类只需实现两个与数据格式相关的抽象方法：
   - ``all_names()``：数据集中所有带标签样本的名字；
   - ``load_sample(name)``：读盘并返回上述字典。
   ``__getitem__`` 在返回前统一调用 ``check_sample`` 校验 dtype / 形状 / 取值范围，
   新适配器写错（比如 label 忘了二值化成 {0,1}、image 忘了除以 255）会立刻报错，而不是
   悄悄地让指标出错。

3. 不继承 torch.utils.data.Dataset。这一层返回的是整幅 numpy 图像，经典方法（Frangi、
   匹配滤波）也要用，没必要依赖 torch。U-Net 训练时再在上面包一层"随机 patch 采样"的
   torch Dataset（任务 1.4），职责分离：本层只管"读对数据"。

4. 划分逻辑可复用。``split_names`` 是纯函数：输入样本名、种子、各 split 大小，输出互不相交的
   划分。先对名字排序再用种子打乱，所以结果只取决于（名字集合, 种子, 大小），与文件系统
   列目录的顺序无关；使用 numpy 独立的 ``default_rng(seed)``，不受全局随机状态影响——
   同一种子在任何机器上、任何时候都得到同一划分。
   各 split 大小之和必须**恰好等于**样本数：若小于样本数，多出来的样本会被静默丢弃，
   这通常是 config 写错（比如漏改了某个 size），所以默认直接报错；确实只想用一部分样本时，
   需显式传 ``allow_unused=True``。

5. 校验用 ``raise ValueError`` 而不是 ``assert``。``python -O`` 会把所有 assert 语句删掉，
   用 assert 做数据校验等于"优化模式下校验消失"；assert 只适合表达"程序员自己的内部假设"。
   数据格式不对属于输入错误，应抛出明确的异常类型。
"""

from abc import ABC, abstractmethod

import numpy as np

SPLITS = ("train", "val", "test")


def split_names(names, seed: int, sizes: dict, allow_unused: bool = False) -> dict:
    """把样本名确定性地划分为 train / val / test。

    参数
    ----
    names : 可迭代的样本名（顺序无关，内部会排序）
    seed  : 划分种子
    sizes : 形如 {"train": 12, "val": 4, "test": 4}；每项为非负整数，总和必须等于样本数
    allow_unused : 为 True 时允许总和小于样本数（多余样本不进入任何 split）；默认 False 时报错

    返回
    ----
    {"train": [...], "val": [...], "test": [...]}，每个列表内部按名字排序，便于阅读。
    """
    names = sorted(names)
    if len(set(names)) != len(names):
        raise ValueError("样本名有重复")
    for s in SPLITS:
        if s not in sizes:
            raise ValueError(f"sizes 缺少 split {s!r}")
        if not isinstance(sizes[s], (int, np.integer)) or sizes[s] < 0:
            raise ValueError(f"sizes[{s!r}] 应为非负整数，实际 {sizes[s]!r}")
    total = sum(sizes[s] for s in SPLITS)
    if total > len(names):
        raise ValueError(f"划分大小之和 {total} 超过样本数 {len(names)}")
    if total < len(names) and not allow_unused:
        raise ValueError(f"划分大小之和 {total} 少于样本数 {len(names)}，"
                         f"有 {len(names) - total} 个样本不会被使用；若是有意为之，请传 allow_unused=True")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    shuffled = [names[i] for i in order]

    result, start = {}, 0
    for s in SPLITS:
        result[s] = sorted(shuffled[start:start + sizes[s]])
        start += sizes[s]
    return result


def check_sample(sample: dict) -> None:
    """校验一个样本是否符合基类契约；不符合时抛出 ValueError，指明哪里错了。"""
    for key in ("image", "label", "fov", "name"):
        if key not in sample:
            raise ValueError(f"样本缺少键 {key!r}")

    image, label, fov = sample["image"], sample["label"], sample["fov"]
    if image.dtype != np.float32:
        raise ValueError(f"image 应为 float32，实际 {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"image 应为 H×W 或 H×W×C，实际形状 {image.shape}")
    if not (0.0 <= image.min() and image.max() <= 1.0):
        raise ValueError(f"image 取值应在 [0, 1]，实际 [{image.min()}, {image.max()}]")

    for key, arr in (("label", label), ("fov", fov)):
        if arr.dtype != np.uint8:
            raise ValueError(f"{key} 应为 uint8，实际 {arr.dtype}")
        if arr.shape != image.shape[:2]:
            raise ValueError(f"{key} 形状 {arr.shape} 与 image {image.shape[:2]} 不一致")
        if arr.max() > 1:
            raise ValueError(f"{key} 取值应为 {{0, 1}}，实际最大值 {arr.max()}")


class VesselDataset(ABC):
    """血管分割数据集抽象基类。子类实现 ``all_names`` 和 ``load_sample`` 即可。

    参数
    ----
    split : "train" / "val" / "test"
    seed  : 划分种子（来自 config）
    sizes : 各 split 样本数（来自 config）
    """

    def __init__(self, split: str, seed: int, sizes: dict):
        if split not in SPLITS:
            raise ValueError(f"split 必须是 {SPLITS} 之一，实际 {split!r}")
        self.split = split
        self.seed = seed
        self.sizes = dict(sizes)
        self.splits = split_names(self.all_names(), seed, self.sizes)  # 三个 split 都存下，便于检查/打印
        self.names = self.splits[split]

    @abstractmethod
    def all_names(self) -> list:
        """返回所有可参与划分的（带标签的）样本名。"""

    @abstractmethod
    def load_sample(self, name: str) -> dict:
        """读取一个样本，返回 {image, label, fov, name} 字典（格式见模块文档）。"""

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict:
        sample = self.load_sample(self.names[index])
        check_sample(sample)
        return sample

    def __repr__(self) -> str:
        return f"{type(self).__name__}(split={self.split!r}, n={len(self)}, seed={self.seed})"

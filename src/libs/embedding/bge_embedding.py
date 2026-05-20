"""BGE 本地 Embedding 实现模块。

使用 FlagEmbedding 加载本地 BGE-M3 模型，支持稠密向量、稀疏向量和 ColBERT 向量。
模型路径通过配置文件指定，默认为 models/bge-m3。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import register_embedding


@register_embedding("bge")
class BGEEmbedding(BaseEmbedding):
    """BGE 本地 Embedding 实现。

    通过 FlagEmbedding 加载本地 BGE-M3 模型，支持：
    - 稠密向量（dense）：语义级别的连续表示
    - 稀疏向量（sparse）：类似 BM25 的词袋表示，适合混合检索
    - ColBERT 向量：多向量表示，适合精细匹配

    延迟加载模型，首次调用 embed 时才加载到内存。
    """

    def __init__(
        self,
        model: str = "bge-m3",
        dimensions: int = 1024,
        api_key: str = "",
        model_path: str = "",
        use_fp16: bool = True,
        **kwargs,
    ):
        """初始化 BGE Embedding 实例。

        参数:
            model: 模型名称（如 "bge-m3"）
            dimensions: 输出向量维度（bge-m3 默认 1024）
            api_key: 未使用，保留接口一致性
            model_path: 本地模型路径（如 "models/bge-m3"）
            use_fp16: 是否使用半精度推理（默认 True，GPU 下更快）
            **kwargs: 其他参数
        """
        super().__init__(model, dimensions, api_key, model_path, **kwargs)
        self.use_fp16 = use_fp16
        self._embedding_model = None  # 延迟加载

    def _load_model(self):
        """延迟加载 FlagEmbedding 模型。

        首次调用 embed 时加载模型，避免启动时占用过多内存。
        先检查路径是否存在，再导入 FlagEmbedding（避免不必要的 C 扩展加载）。

        异常:
            ImportError: 未安装 FlagEmbedding 时抛出
            FileNotFoundError: 模型路径不存在时抛出
            RuntimeError: 模型加载失败时抛出
        """
        if self._embedding_model is not None:
            return

        # 先确定并检查模型路径（避免不必要的 FlagEmbedding 导入）
        model_path = self.model_path
        if not model_path:
            model_path = self.model

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"BGE 模型路径不存在: {model_path}，"
                f"请确保模型已下载到指定位置。"
            )

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise ImportError(
                f"FlagEmbedding 导入失败: {e}\n"
                f"请确保已安装: uv pip install FlagEmbedding"
            ) from e

        try:
            self._embedding_model = BGEM3FlagModel(
                model_path,
                use_fp16=self.use_fp16,
            )
        except Exception as e:
            if isinstance(e, (ImportError, FileNotFoundError)):
                raise
            raise RuntimeError(
                f"BGE 模型加载失败 (path={model_path}): {e}"
            ) from e

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """批量将文本转换为稠密向量。

        参数:
            texts: 待嵌入的文本列表
            **kwargs: 额外参数

        返回:
            向量列表，每个向量是浮点数列表，长度等于 dimensions

        异常:
            ValueError: 空输入时抛出
        """
        if not texts:
            raise ValueError("BGE Embedding 输入文本列表不能为空")

        self._load_model()

        try:
            output = self._embedding_model.encode(
                texts,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            return [vec.tolist() for vec in output["dense_vecs"]]
        except Exception as e:
            raise RuntimeError(
                f"BGE Embedding 编码失败 (model={self.model}): {e}"
            ) from e

    def embed_with_sparse(
        self, texts: List[str], **kwargs
    ) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """批量编码，同时返回稠密向量和稀疏向量。

        参数:
            texts: 待嵌入的文本列表

        返回:
            (dense_vecs, sparse_vecs) 元组：
            - dense_vecs: 稠密向量列表
            - sparse_vecs: 稀疏向量列表，每个元素是 {token_id: weight} 字典
        """
        if not texts:
            raise ValueError("BGE Embedding 输入文本列表不能为空")

        self._load_model()

        try:
            output = self._embedding_model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense = [vec.tolist() for vec in output["dense_vecs"]]
            sparse = output["lexical_weights"]
            return dense, sparse
        except Exception as e:
            raise RuntimeError(
                f"BGE Embedding 编码失败 (model={self.model}): {e}"
            ) from e

    def embed_with_colbert(
        self, texts: List[str], **kwargs
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        """批量编码，同时返回稠密向量和 ColBERT 向量。

        参数:
            texts: 待嵌入的文本列表

        返回:
            (dense_vecs, colbert_vecs) 元组：
            - dense_vecs: 稠密向量列表
            - colbert_vecs: ColBERT 向量列表，每个元素是 (token_num, dim) 的二维列表
        """
        if not texts:
            raise ValueError("BGE Embedding 输入文本列表不能为空")

        self._load_model()

        try:
            output = self._embedding_model.encode(
                texts,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=True,
            )
            dense = [vec.tolist() for vec in output["dense_vecs"]]
            colbert = [cv.tolist() for cv in output["colbert_vecs"]]
            return dense, colbert
        except Exception as e:
            raise RuntimeError(
                f"BGE Embedding 编码失败 (model={self.model}): {e}"
            ) from e

    def embed_all(
        self, texts: List[str], **kwargs
    ) -> Dict[str, list]:
        """批量编码，返回所有三种向量。

        参数:
            texts: 待嵌入的文本列表

        返回:
            字典，包含：
            - "dense": 稠密向量列表
            - "sparse": 稀疏向量列表
            - "colbert": ColBERT 向量列表
        """
        if not texts:
            raise ValueError("BGE Embedding 输入文本列表不能为空")

        self._load_model()

        try:
            output = self._embedding_model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
            )
            return {
                "dense": [vec.tolist() for vec in output["dense_vecs"]],
                "sparse": output["lexical_weights"],
                "colbert": [cv.tolist() for cv in output["colbert_vecs"]],
            }
        except Exception as e:
            raise RuntimeError(
                f"BGE Embedding 编码失败 (model={self.model}): {e}"
            ) from e

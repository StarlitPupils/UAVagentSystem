# core/memory/embedding.py
"""文本嵌入服务 - 优先使用镜像站，自动降级"""
import os
import numpy as np
from config.settings import config

class EmbeddingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self.model_name = config.EMBEDDING_MODEL
        self.dimension = 384

        # 解决 SSL / 网络问题：优先使用镜像站
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFY", "1")
        os.environ.setdefault("CURL_CA_BUNDLE", "")
        os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
        # 禁用 huggingface_hub 的 SSL 验证
        os.environ.setdefault("HF_HUB_ENABLE_DOWNLOAD_RESUME", "0")

    def _lazy_load(self):
        if self.model is not None:
            return
        try:
            # 强制禁用 SSL 验证（部分环境需要）
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            
            from sentence_transformers import SentenceTransformer
            # 使用本地缓存优先，避免重复下载
            self.model = SentenceTransformer(
                self.model_name,
                device="cpu",
                trust_remote_code=True,
                cache_folder=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            )
            self.dimension = getattr(self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension)()
            print(f"[Embedding] 模型 {self.model_name} 加载完成，维度={self.dimension}")
        except Exception as e:
            print(f"[Embedding] 模型加载失败（将使用哈希降级）: {e}")
            self.model = None
            self.dimension = 384

    def encode(self, texts: list[str]) -> np.ndarray:
        self._lazy_load()
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, show_progress_bar=False)
                return np.array(embeddings)
            except Exception as e:
                print(f"[Embedding] 编码失败，降级到哈希: {e}")
        return self._hash_fallback(texts)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _hash_fallback(self, texts: list[str]) -> np.ndarray:
        """SHA256 哈希降级（384维，归一化）"""
        import hashlib
        dim = 384
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
            for j in range(dim):
                vectors[i, j] = hash_bytes[j % len(hash_bytes)] / 255.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-8)
        return vectors

embedding_service = EmbeddingService()

# core/memory/vector_store.py
"""ChromaDB 向量存储 - 绕过 numpy 维度陷阱，静默降级"""
import os
import json
import uuid
from datetime import datetime
from config.settings import config

class VectorStore:
    def __init__(self, collection_name: str = "uavagent_memory"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialized = False
        self._use_fallback = True
        # 显式保证是 Python 原生 int，永远不用 numpy 整数
        self.embedding_dim = 384

    def initialize(self):
        if self._initialized:
            return

        # 安全获取维度（强制转为原生 int）
        try:
            from .embedding import embedding_service
            embedding_service._lazy_load()
            self.embedding_dim = int(embedding_service.dimension)  # ← 强制 int
        except Exception:
            self.embedding_dim = 384

        # 尝试 ChromaDB
        try:
            import chromadb

            db_path = config.CHROMA_DB_PATH
            os.makedirs(db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=db_path)

            # 先获取已有 collection，若不存在则创建
            try:
                self.collection = self.client.get_collection(self.collection_name)
                # 检查维度是否匹配（如果已有数据）
                cnt = self.collection.count()
                if cnt > 0:
                    sample = self.collection.get(limit=1, include=["embeddings"])
                    if sample and "embeddings" in sample and sample["embeddings"]:
                        first_emb = sample["embeddings"][0]
                        # 安全获取已有维度
                        if hasattr(first_emb, '__len__'):
                            old_dim = len(first_emb)
                        else:
                            old_dim = 384
                        # 转换为原生 int 再比较
                        old_dim = int(old_dim)
                        cur_dim = int(self.embedding_dim)
                        if old_dim != cur_dim:
                            print(f"[VectorStore] 维度变更 {old_dim}→{cur_dim}，重建")
                            self.client.delete_collection(self.collection_name)
                            self.collection = self.client.create_collection(
                                name=self.collection_name,
                                metadata={"dim": cur_dim}
                            )
            except Exception:
                # collection 不存在，创建
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"dim": int(self.embedding_dim)}
                )

            self._use_fallback = False
            self._initialized = True
            print(f"[VectorStore] ChromaDB 就绪 (条目: {self.collection.count()})")
            return

        except Exception as e:
            # 静默降级，不打印完整错误栈，只给一句提示
            print(f"[VectorStore] ChromaDB 暂不可用，使用 JSON 降级存储")
            self._use_fallback = True
            self._initialized = True

    # ==================== 添加 / 搜索 ====================
    def add_memory(self, content, metadata=None, memory_type="general"):
        mem_id = str(uuid.uuid4())[:8]
        meta = metadata or {}
        meta.update({"timestamp": datetime.now().isoformat(), "type": memory_type})

        if not self._use_fallback and self.collection is not None:
            try:
                from .embedding import embedding_service
                emb = embedding_service.encode_single(content).tolist()
                self.collection.add(ids=[mem_id], embeddings=[emb],
                                    documents=[content], metadatas=[meta])
                return mem_id
            except Exception:
                pass

        self._json_add(mem_id, content, meta)
        return mem_id

    def search(self, query, top_k=5, memory_type=None):
        results = []

        if not self._use_fallback and self.collection is not None:
            try:
                from .embedding import embedding_service
                q_emb = embedding_service.encode_single(query).tolist()
                where = {"type": memory_type} if memory_type else None
                cr = self.collection.query(
                    query_embeddings=[q_emb],
                    n_results=top_k,
                    where=where,
                    include=["documents", "metadatas", "distances"]
                )
                if cr and cr.get("ids") and cr["ids"][0]:
                    for i in range(len(cr["ids"][0])):
                        results.append({
                            "id": cr["ids"][0][i],
                            "content": cr["documents"][0][i],
                            "metadata": cr["metadatas"][0][i],
                            "score": 1.0 - min(cr["distances"][0][i], 1.0)
                        })
                return results
            except Exception:
                pass

        return self._json_search(query, top_k, memory_type)

    # ==================== JSON 降级 ====================
    def _fallback_path(self):
        d = getattr(config, 'OUTPUT_DIR', None) or "output"
        return os.path.join(d, "memory_fallback.json")

    def _json_add(self, mem_id, content, meta):
        path = self._fallback_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        records = []
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    records = json.load(f)
            except Exception:
                records = []
        records.append({"id": mem_id, "content": content, "metadata": meta})
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(records[-500:], f, indent=2, ensure_ascii=False)

    def _json_search(self, query, top_k, memory_type=None):
        path = self._fallback_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except Exception:
            return []
        qw = set(query.lower().split())
        scored = []
        for r in records:
            if memory_type and r.get("metadata", {}).get("type") != memory_type:
                continue
            cw = set(r["content"].lower().split())
            score = len(qw & cw) / max(len(qw), 1)
            if score > 0:
                scored.append({**r, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def clear(self):
        if not self._use_fallback and self.collection is not None:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"dim": int(self.embedding_dim)}
                )
            except Exception:
                pass

vector_store = VectorStore()

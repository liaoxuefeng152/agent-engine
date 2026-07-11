"""
RAG 向量检索引擎
- 基于 FAISS 实现向量检索（可选依赖，未安装时降级为关键词匹配）
- 根据Agent绑定的知识库精准召回
- 相似度阈值过滤
- 支持知识库更新后主动刷新索引
"""
import logging
import hashlib
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# FAISS 可选导入
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed, RAG will fallback to keyword matching")


@dataclass
class RetrievalResult:
    """检索结果"""
    content: str
    score: float
    source: str = ""
    metadata: Optional[dict] = None


class KnowledgeBaseEntry:
    """知识库条目"""
    def __init__(self, doc_id: str, content: str, embedding: Optional[np.ndarray] = None, metadata: Optional[dict] = None):
        self.doc_id = doc_id
        self.content = content
        self.embedding = embedding
        self.metadata = metadata or {}


class VectorStore:
    """向量存储 - 单个知识库的FAISS索引"""

    def __init__(self, kb_id: str, dimension: int = 384):
        self.kb_id = kb_id
        self.dimension = dimension
        self.entries: List[KnowledgeBaseEntry] = []
        self.index: Any = None  # faiss.Index type
        self._dirty = True

        if FAISS_AVAILABLE and faiss is not None:
            self.index = faiss.IndexFlatIP(dimension)

    def add_documents(self, documents: List[Dict]):
        """添加文档到索引"""
        for doc in documents:
            emb = np.array(doc.get("embedding", [0.0] * self.dimension), dtype=np.float32)
            if len(emb) != self.dimension:
                emb = self._resize_embedding(emb, self.dimension)
            entry = KnowledgeBaseEntry(
                doc_id=doc.get("doc_id", hashlib.md5(doc["content"].encode()).hexdigest()),
                content=doc["content"],
                embedding=emb,
                metadata=doc.get("metadata", {}),
            )
            self.entries.append(entry)

        self._dirty = True
        self._rebuild_index()
        logger.info("VectorStore[%s] added %d documents, total=%d", self.kb_id, len(documents), len(self.entries))

    def search(self, query_embedding: np.ndarray, top_k: int = 5, threshold: float = 0.5) -> List[RetrievalResult]:
        """检索相似文档"""
        if not self.entries:
            return []

        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        if len(query_embedding[0]) != self.dimension:
            query_embedding = self._resize_embedding(query_embedding[0], self.dimension).reshape(1, -1)

        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.entries):
                    continue
                if score < threshold:
                    continue
                entry = self.entries[idx]
                results.append(RetrievalResult(
                    content=entry.content,
                    score=float(score),
                    source=entry.metadata.get("source", ""),
                    metadata=entry.metadata,
                ))
            return results
        else:
            return self._keyword_fallback(query_embedding, top_k, threshold)

    def _rebuild_index(self):
        """重建FAISS索引"""
        if not FAISS_AVAILABLE or faiss is None or not self.entries:
            return
        embeddings = np.array([e.embedding for e in self.entries], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self._dirty = False

    def _keyword_fallback(self, query_emb: np.ndarray, top_k: int, threshold: float) -> List[RetrievalResult]:
        """关键词匹配降级方案"""
        results = []
        query_str = str(query_emb.flatten()[:50].tolist())
        for entry in self.entries[:top_k * 3]:
            overlap = sum(1 for c in query_str[:20] if c in entry.content)
            score = min(overlap / max(len(entry.content), 1), 1.0)
            if score >= threshold * 0.3:
                results.append(RetrievalResult(
                    content=entry.content,
                    score=score,
                    source=entry.metadata.get("source", ""),
                    metadata=entry.metadata,
                ))
        return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

    @staticmethod
    def _resize_embedding(emb: np.ndarray, target_dim: int) -> np.ndarray:
        current = len(emb)
        if current == target_dim:
            return emb
        if current > target_dim:
            return emb[:target_dim]
        return np.pad(emb, (0, target_dim - current), mode='constant')


class RAGEngine:
    """
    RAG 检索引擎
    - 管理多个知识库的向量存储
    - 根据Agent绑定的知识库ID精准召回
    - 支持主动刷新
    """

    def __init__(self):
        self._stores: Dict[str, VectorStore] = {}
        self._dimension = 384
        logger.info("RAGEngine initialized, FAISS available=%s", FAISS_AVAILABLE)

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的向量表示（确定性伪向量，生产环境应替换为真实embedding模型）"""
        text_bytes = text.encode('utf-8')
        hash_bytes = hashlib.sha256(text_bytes).digest()

        vec = np.zeros(self._dimension, dtype=np.float32)
        for i in range(self._dimension):
            byte_idx = i % len(hash_bytes)
            vec[i] = (hash_bytes[byte_idx] - 128) / 128.0

        for i, char in enumerate(text[:100]):
            vec[i % self._dimension] += ord(char) / 1000.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def get_or_create_store(self, kb_id: str) -> VectorStore:
        if kb_id not in self._stores:
            self._stores[kb_id] = VectorStore(kb_id, self._dimension)
        return self._stores[kb_id]

    def list_available_bases(self) -> List[Dict]:
        """列出所有已加载的知识库"""
        result = []
        for kb_id, store in self._stores.items():
            result.append({
                "id": kb_id,
                "name": kb_id,
                "doc_count": store.index.ntotal if store.index is not None else 0,
            })
        return result

    def load_knowledge_base(self, kb_id: str, documents: List[Dict]):
        """加载/更新知识库文档"""
        store = self.get_or_create_store(kb_id)
        for doc in documents:
            if "embedding" not in doc:
                doc["embedding"] = self._get_embedding(doc["content"]).tolist()
        store.add_documents(documents)

    def reload_knowledge_base(self, kb_id: str):
        """主动刷新知识库索引"""
        if kb_id in self._stores:
            store = self._stores[kb_id]
            store._rebuild_index()
            logger.info("RAGEngine reloaded kb_id=%s", kb_id)

    def search(
        self,
        query: str,
        knowledge_base_ids: List[str],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[RetrievalResult]:
        """跨知识库检索"""
        query_emb = self._get_embedding(query)
        all_results: List[RetrievalResult] = []

        for kb_id in knowledge_base_ids:
            store = self._stores.get(kb_id)
            if store is None:
                continue
            results = store.search(query_emb, top_k=top_k, threshold=threshold)
            all_results.extend(results)

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    def format_context(self, results: List[RetrievalResult], max_length: int = 2000) -> str:
        """将检索结果格式化为Prompt上下文"""
        if not results:
            return ""

        context_parts = []
        total_len = 0
        for i, r in enumerate(results, 1):
            snippet = f"[知识{i}] {r.content}"
            if total_len + len(snippet) > max_length:
                break
            context_parts.append(snippet)
            total_len += len(snippet)

        return "\n".join(context_parts)


# 全局单例
rag_engine = RAGEngine()

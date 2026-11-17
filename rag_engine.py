import json
import re
from typing import List, Dict, Any, Tuple
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import ollama


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class EnterpriseRAGEngine:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        rerank_model_name: str = "BAAI/bge-reranker-base",
        llm_model: str = "llama3.1",
    ):
        self.llm_model = llm_model
        self.embed_model = SentenceTransformer(embed_model_name)
        self.rerank_model = CrossEncoder(rerank_model_name)

        # Persistent Vector Database
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name="enterprise_knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25 = None
        self.corpus_docs: List[str] = []
        self.corpus_metas: List[Dict[str, Any]] = []
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self) -> None:
        """Synchronizes in-memory BM25 keyword index with persistent ChromaDB."""
        existing = self.collection.get(include=["documents", "metadatas"])
        self.corpus_docs = existing.get("documents") or []
        self.corpus_metas = existing.get("metadatas") or []
        if self.corpus_docs:
            tokenized_corpus = [_tokenize(doc) for doc in self.corpus_docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def ingest_pdfs(
        self, pdf_files: List[Any], chunk_size: int = 600, chunk_overlap: int = 100
    ) -> int:
        """Parses PDFs, chunks with page metadata, embeds, and indexes into Chroma + BM25."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs, metas, ids = [], [], []
        base_id = self.collection.count()

        for file_obj in pdf_files:
            file_name = getattr(file_obj, "name", str(file_obj))
            reader = PdfReader(file_obj)
            for page_idx, page in enumerate(reader.pages, start=1):
                raw_text = (page.extract_text() or "").strip()
                if not raw_text:
                    continue
                for chunk in splitter.split_text(raw_text):
                    docs.append(chunk)
                    metas.append({"source": file_name, "page": page_idx})
                    ids.append(f"doc_{base_id + len(docs)}")

        if docs:
            embeddings = self.embed_model.encode(docs, normalize_embeddings=True).tolist()
            self.collection.add(
                documents=docs, embeddings=embeddings, metadatas=metas, ids=ids
            )
            self._rebuild_bm25_index()
        return len(docs)

    def ingest_raw_texts(
        self, records: List[Dict[str, Any]], chunk_size: int = 600, chunk_overlap: int = 100
    ) -> int:
        """Used by evaluation notebooks to index benchmark corpora directly."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs, metas, ids = [], [], []
        base_id = self.collection.count()

        for rec in records:
            for chunk in splitter.split_text(rec["text"]):
                docs.append(chunk)
                metas.append(
                    {"source": rec.get("source", "benchmark_doc"), "page": rec.get("page", 1)}
                )
                ids.append(f"bench_{base_id + len(docs)}")

        if docs:
            embeddings = self.embed_model.encode(docs, normalize_embeddings=True).tolist()
            self.collection.add(
                documents=docs, embeddings=embeddings, metadatas=metas, ids=ids
            )
            self._rebuild_bm25_index()
        return len(docs)

    def reset_database(self) -> None:
        self.chroma_client.delete_collection("enterprise_knowledge_base")
        self.collection = self.chroma_client.get_or_create_collection(
            name="enterprise_knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25 = None
        self.corpus_docs = []
        self.corpus_metas = []

    def retrieve(
        self,
        query: str,
        top_k_candidates: int = 12,
        top_k_final: int = 4,
        mode: str = "hybrid_rerank",
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Supports three ablation modes:
        - 'dense': Pure vector similarity search
        - 'hybrid': Dense + BM25 fused via Reciprocal Rank Fusion (RRF)
        - 'hybrid_rerank': Hybrid RRF followed by Cross-Encoder reranking
        """
        total_docs = len(self.corpus_docs)
        if total_docs == 0:
            return []

        k_fetch = min(top_k_candidates, total_docs)

        # 1. Dense Retrieval
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).tolist()
        dense_res = self.collection.query(query_embeddings=q_emb, n_results=k_fetch)
        dense_docs = dense_res["documents"][0]
        dense_metas = dense_res["metadatas"][0]

        if mode == "dense":
            return [
                {
                    "text": doc,
                    "meta": meta,
                    "dense_rank": idx + 1,
                    "bm25_rank": None,
                    "rrf_score": 1.0 / (idx + 1),
                    "rerank_score": None,
                }
                for idx, (doc, meta) in enumerate(zip(dense_docs[:top_k_final], dense_metas[:top_k_final]))
            ]

        # 2. Sparse BM25 Retrieval + Reciprocal Rank Fusion (RRF)
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_top_indices = sorted(
            range(total_docs), key=lambda i: bm25_scores[i], reverse=True
        )[:k_fetch]

        fused: Dict[str, Dict[str, Any]] = {}

        for rank, (doc, meta) in enumerate(zip(dense_docs, dense_metas), start=1):
            fused[doc] = {
                "text": doc,
                "meta": meta,
                "dense_rank": rank,
                "bm25_rank": None,
                "rrf_score": alpha * (1.0 / (rrf_k + rank)),
                "rerank_score": None,
            }

        for rank, doc_idx in enumerate(bm25_top_indices, start=1):
            doc = self.corpus_docs[doc_idx]
            meta = self.corpus_metas[doc_idx]
            if doc not in fused:
                fused[doc] = {
                    "text": doc,
                    "meta": meta,
                    "dense_rank": None,
                    "bm25_rank": rank,
                    "rrf_score": 0.0,
                    "rerank_score": None,
                }
            fused[doc]["bm25_rank"] = rank
            fused[doc]["rrf_score"] += (1.0 - alpha) * (1.0 / (rrf_k + rank))

        hybrid_ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:k_fetch]

        if mode == "hybrid":
            return hybrid_ranked[:top_k_final]

        # 3. Cross-Encoder Reranking
        pairs = [[query, item["text"]] for item in hybrid_ranked]
        ce_scores = self.rerank_model.predict(pairs)
        for item, score in zip(hybrid_ranked, ce_scores):
            item["rerank_score"] = float(score)

        reranked = sorted(hybrid_ranked, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k_final]

    def generate_answer(
        self, query: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Generates a strictly grounded response with [1], [2] inline citations."""
        context_lines = []
        for idx, item in enumerate(retrieved_chunks, start=1):
            src = item["meta"].get("source", "Unknown")
            page = item["meta"].get("page", "?")
            context_lines.append(f"[{idx}] (Source: {src}, Page {page}):\n{item['text']}")

        context_str = "\n\n".join(context_lines)

        system_prompt = (
            "You are a rigorous Enterprise AI Knowledge Assistant. "
            "Answer the user's question using ONLY the numbered context passages below. "
            "Every factual claim MUST end with an inline citation like [1] or [1][2]. "
            "If the context does not contain enough information to answer, reply strictly with: "
            "'I cannot verify this from the uploaded documents.'"
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"

        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.1},
        )
        return response["message"]["content"], context_str

    def evaluate_hallucination(self, answer: str, context_str: str) -> Dict[str, Any]:
        """
        LLM-as-a-Judge Faithfulness & Claim Verification:
        Decomposes the answer into atomic claims and checks each against retrieved context.
        """
        eval_prompt = f"""You are an automated Hallucination & Faithfulness Auditor.
Break the Answer into individual factual claims and verify whether each claim is directly supported by the Context.

Return ONLY a valid JSON object matching this exact schema:
{{
  "faithfulness_score": <float between 0.0 and 1.0>,
  "verdict": "<'Faithful' if score >= 0.85 else 'Partial Hallucination' if score >= 0.5 else 'Hallucinated'>",
  "claims": [
    {{"claim": "<atomic claim text>", "supported": <true or false>, "evidence": "<short quote or 'Not found in context'>"}}
  ],
  "summary": "<one concise sentence explaining the audit result>"
}}

Context:
{context_str}

Answer:
{answer}"""

        res = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": eval_prompt}],
            format="json",
            options={"temperature": 0.0},
        )
        try:
            return json.loads(res["message"]["content"])
        except Exception:
            return {
                "faithfulness_score": 0.0,
                "verdict": "Audit Parse Error",
                "claims": [],
                "summary": res["message"]["content"],
            }

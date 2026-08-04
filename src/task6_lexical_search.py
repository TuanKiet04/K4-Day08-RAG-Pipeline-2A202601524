"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

# TODO: Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
BM25_INSTANCE = None

def init_corpus_and_bm25():
    """Tải dữ liệu từ vector store (để đồng bộ) và khởi tạo BM25"""
    global CORPUS, BM25_INSTANCE
    if CORPUS and BM25_INSTANCE is not None:
        return
        
    import chromadb
    try:
        from src.task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME
    except ModuleNotFoundError:
        from task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME
    
    # Lấy chunks từ ChromaDB đã lưu ở Task 4
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    results = collection.get()
    
    if results and results.get("documents"):
        for i in range(len(results["ids"])):
            CORPUS.append({
                "content": results["documents"][i],
                "metadata": results["metadatas"][i] if results.get("metadatas") else {}
            })
            
    if not CORPUS:
        # Fallback lấy từ markdown nếu Chroma rỗng (chưa chạy task 4)
        try:
            from src.task4_chunking_indexing import load_documents, chunk_documents
        except ModuleNotFoundError:
            from task4_chunking_indexing import load_documents, chunk_documents
        CORPUS = chunk_documents(load_documents())
        
    BM25_INSTANCE = build_bm25_index(CORPUS)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi
    
    # Tokenize - Tách từ đơn giản bằng khoảng trắng (có thể dùng underthesea để chia từ tiếng Việt tốt hơn)
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    # Khởi tạo dữ liệu và mô hình (chỉ chạy lần đầu)
    init_corpus_and_bm25()
    
    if not CORPUS or not BM25_INSTANCE:
        return []

    # Tokenize câu truy vấn
    tokenized_query = query.lower().split()
    scores = BM25_INSTANCE.get_scores(tokenized_query)
    
    # Lấy top_k kết quả có điểm cao nhất
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0: # Chỉ lấy các đoạn có chứa ít nhất 1 từ khoá
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
            
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

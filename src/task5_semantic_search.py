"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import os
    import sys
    from pathlib import Path
    
    # Đảm bảo import được module từ cùng thư mục
    sys.path.append(str(Path(__file__).parent))
    from task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
    
    import chromadb
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    
    # 1. Khởi tạo official OpenAI client để tính vector cho query
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Thiếu OPENAI_API_KEY trong .env để tạo query embedding")
    client_oai = OpenAI(api_key=api_key)
    
    response = client_oai.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    query_vector = response.data[0].embedding
    
    # 2. Khởi tạo kết nối tới database ChromaDB
    client_chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client_chroma.get_collection(name=COLLECTION_NAME)
    
    # 3. Tìm kiếm bằng vector similarity
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    
    # 4. Format kết quả trả về
    output = []
    if results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            # ChromaDB lưu distance theo cosine. distance = 1 - similarity.
            # Do đó score (độ tương đồng) = 1.0 - distance
            score = max(0.0, 1.0 - dist)
            output.append({"content": doc, "score": round(score, 4), "metadata": meta})
    
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    # Test
    print("="*50)
    print("TEST: Task 5 - Semantic Search")
    print("="*50)
    query = "Quy định đổi trả hàng bị lỗi của Tiki"
    print(f"Query: '{query}'\n")
    results = semantic_search(query, top_k=3)
    
    for i, r in enumerate(results):
        print(f"--- Top {i+1} [Score: {r['score']:.4f}] ---")
        print(f"File: {r['metadata'].get('source', 'N/A')}")
        print(f"Nội dung: {r['content'][:150]}...\n")

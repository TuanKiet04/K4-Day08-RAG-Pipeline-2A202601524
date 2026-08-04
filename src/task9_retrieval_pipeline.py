"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:  # Cho phép chạy trực tiếp: python src/task9_retrieval_pipeline.py
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# TODO: Calibrate threshold này bằng cách tự đo điểm cosine của semantic_search
# cho câu hỏi liên quan vs câu hỏi lạc đề (xem ghi chú ở trên) — ĐỪNG copy nguyên
# giá trị mẫu, mỗi corpus/embedding model sẽ cho khoảng điểm khác nhau.
SCORE_THRESHOLD = 0.3   # Nếu best score (cosine gốc) < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def _normalize_result(item: dict, retriever_name: str) -> dict | None:
    """Chuẩn hóa output từ các retriever để Task 9 xử lý thống nhất."""
    if not isinstance(item, dict) or not item.get("content"):
        return None

    try:
        score = float(item.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0

    metadata = item.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}

    normalized = {
        "content": str(item["content"]),
        "score": score,
        "metadata": metadata.copy(),
    }

    # Lưu điểm gốc trong metadata để debug/eval, không dùng làm source hiển thị.
    normalized["metadata"][f"{retriever_name}_score"] = score
    return normalized


def _safe_search(search_fn, name: str, query: str, top_k: int) -> list[dict]:
    """
    Gọi retriever và trả list chuẩn hóa.

    Task 5/6 phụ thuộc ChromaDB, embedding API và index đã được build. Trong lúc demo
    hoặc chấm test có thể thiếu một trong các điều kiện đó, nên Task 9 không được crash;
    nếu retriever lỗi thì pipeline vẫn có thể thử fallback PageIndex.
    """
    try:
        raw_results = search_fn(query, top_k=top_k) or []
    except Exception as exc:
        print(f"  [warn] {name} failed: {exc}")
        return []

    normalized = []
    for item in raw_results:
        result = _normalize_result(item, name)
        if result is not None:
            normalized.append(result)
    return normalized


def _mark_hybrid(results: list[dict]) -> list[dict]:
    """Gắn source retrieval-level mà không đụng metadata['source'] của tài liệu."""
    marked = []
    for item in results:
        copied = item.copy()
        copied["metadata"] = (item.get("metadata") or {}).copy()
        copied["source"] = "hybrid"
        marked.append(copied)
    return marked


def _fallback_pageindex(query: str, top_k: int) -> list[dict]:
    """Gọi PageIndex fallback và chuẩn hóa source marker."""
    try:
        fallback = pageindex_search(query, top_k=top_k) or []
    except Exception as exc:
        print(f"  [warn] pageindex fallback failed: {exc}")
        return []

    results = []
    for item in fallback[:top_k]:
        normalized = _normalize_result(item, "pageindex")
        if normalized is None:
            continue
        normalized["source"] = "pageindex"
        results.append(normalized)
    return results


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if top_k <= 0:
        return []

    candidate_k = max(top_k * 2, top_k)

    # Step 1: chạy semantic + lexical. Hai retriever độc lập nên có thể chạy song song.
    try:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                _safe_search, semantic_search, "semantic", query, candidate_k
            )
            sparse_future = executor.submit(
                _safe_search, lexical_search, "lexical", query, candidate_k
            )
            dense_results = dense_future.result()
            sparse_results = sparse_future.result()
    except Exception as exc:
        # Defensive fallback: nếu ThreadPool hoặc môi trường có vấn đề, chạy tuần tự.
        print(f"  [warn] parallel retrieval failed, retry sequentially: {exc}")
        dense_results = _safe_search(semantic_search, "semantic", query, candidate_k)
        sparse_results = _safe_search(lexical_search, "lexical", query, candidate_k)

    # Step 2: quyết định fallback bằng cosine gốc từ semantic_search, không dùng RRF.
    best_dense_score = dense_results[0]["score"] if dense_results else 0.0
    should_fallback = best_dense_score < score_threshold

    # Step 3: merge dense + sparse bằng RRF để giữ lợi thế hybrid retrieval.
    if dense_results or sparse_results:
        merged = rerank_rrf([dense_results, sparse_results], top_k=candidate_k)
    else:
        merged = []

    merged = _mark_hybrid(merged)

    # Step 4: rerank lần cuối. Với method hiện tại là RRF, thao tác này chủ yếu chuẩn hóa
    # thứ hạng cuối; nếu đổi sang cross-encoder sau này thì cùng interface vẫn dùng được.
    if use_reranking and merged:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
            final_results = _mark_hybrid(final_results)
        except NotImplementedError as exc:
            print(f"  [warn] rerank method '{RERANK_METHOD}' not ready: {exc}")
            final_results = merged[:top_k]
        except Exception as exc:
            print(f"  [warn] rerank failed: {exc}")
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    # Step 5: fallback sang PageIndex nếu semantic confidence thấp hoặc hybrid rỗng.
    if should_fallback or not final_results:
        print(
            f"  [warn] Semantic best score ({best_dense_score:.3f}) "
            f"< threshold ({score_threshold:.3f}); trying PageIndex fallback"
        )
        fallback = _fallback_pageindex(query, top_k=top_k)
        if fallback:
            return fallback[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What payment methods does Shopee support?",
        "How do I request a return or refund?",
        "What evidence do I need for a refund request?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")

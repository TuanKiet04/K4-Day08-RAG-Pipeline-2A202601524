"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# Model sinh câu trả lời. Khi dùng official OpenAI API, model không có prefix "openai/".
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về chính sách thương mại điện tử và hỗ trợ
khách hàng (thanh toán, đổi trả, giao hàng, quyền riêng tư, quy định người bán).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Returns Policy, 2026]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return list(chunks)

    # front = even indices (0, 2, 4...) → đầu prompt
    # back  = odd indices  (1, 3, 5...) → cuối prompt (đảo ngược)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        source = metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type", "unknown")
        score = chunk.get("score", 0.0)
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type} | Score: {score:.4f}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def _build_llm_client():
    """Khởi tạo OpenAI-compatible client.

    Ưu tiên official OpenAI API nếu có OPENAI_API_KEY để tránh lỗi OpenRouter credits
    khi file .env vẫn giữ cả OPENROUTER_API_KEY.
    """
    from openai import OpenAI

    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if provider == "openrouter" and openrouter_key:
        return OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1"), "openrouter"
    if openai_key:
        return OpenAI(api_key=openai_key), "openai"
    if openrouter_key:
        return OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1"), "openrouter"

    raise RuntimeError(
        "Thiếu API key. Hãy set OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong file .env"
    )


def _resolve_llm_model(provider: str) -> str:
    """Chuẩn hóa model id theo provider."""
    model = LLM_MODEL
    if provider == "openai" and model.startswith("openai/"):
        return model.removeprefix("openai/")
    if provider == "openrouter" and "/" not in model:
        return f"openai/{model}"
    return model


def generate_answer_from_chunks(
    query: str,
    chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Sinh câu trả lời từ danh sách chunks đã retrieve sẵn.

    Hàm này phục vụ evaluation A/B: eval có thể tự gọi Task 9 với config khác nhau
    (có rerank / không rerank), rồi dùng chung prompt generation của Task 10.
    """
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có",
            "sources": [],
            "retrieval_source": "none",
        }

    # Reorder để tránh lost-in-the-middle.
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    user_message = (
        f"Context:\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Hãy trả lời ngắn gọn, trực tiếp, chỉ dựa trên Context ở trên. "
        f"Mỗi ý chính cần cite đúng tên file trong trường Source, ví dụ [5.md] hoặc [article_03.md]. "
        f"Không cite dạng [Document 1]. "
        f"Nếu thiếu bằng chứng, nói rõ không thể xác minh."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        for turn in chat_history[-4:]:
            role = turn.get("role")
            content = turn.get("content")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    client, provider = _build_llm_client()
    response = client.chat.completions.create(
        model=_resolve_llm_model(provider),
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=700,
    )
    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có"

    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        top_k: Số chunks retrieval
        chat_history: Lịch sử chat (optional) — list of {'role', 'content'}

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k) or []
    return generate_answer_from_chunks(query, chunks, chat_history=chat_history)


if __name__ == "__main__":
    test_queries = [
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để yêu cầu đổi trả hay hoàn tiền?",
        "Cần chuẩn bị bằng chứng gì khi yêu cầu hoàn tiền?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")

"""
RAG Chatbot — E-commerce Support
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="E-commerce Support RAG Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# HELPERS
# =============================================================================

def render_sources(sources: list[dict], retrieval_source: str | None = None):
    """Hiển thị danh sách chunks đã dùng làm nguồn."""
    if not sources:
        return

    label = f"📚 Nguồn tham khảo ({len(sources)} chunks"
    if retrieval_source:
        label += f" | via `{retrieval_source}`"
    label += ")"

    with st.expander(label):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {}) or {}
            source_name = meta.get("source", "Unknown")
            doc_type = meta.get("type", "unknown")
            score = float(src.get("score", 0) or 0)
            content = src.get("content", "") or ""
            preview = content[:300] + ("..." if len(content) > 300 else "")

            st.markdown(
                f"**[{i}] {source_name}** `{doc_type}` | score: `{score:.4f}`"
            )
            st.text(preview)
            st.divider()


def ask_rag(query: str, top_k: int) -> dict:
    """Gọi Task 10 và trả về dict chuẩn hoá."""
    from src.task10_generation import generate_with_citation

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in {"user", "assistant"}
    ]

    response = generate_with_citation(query, top_k=top_k, chat_history=history)
    return {
        "answer": response.get("answer", "Chưa thể trả lời."),
        "sources": response.get("sources", []) or [],
        "retrieval_source": response.get("retrieval_source", "unknown"),
    }


# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("🛒 E-commerce Support RAG")
    st.caption(
        "Trợ lý hỏi đáp về chính sách thương mại điện tử và hỗ trợ khách hàng "
        "(đổi trả, thanh toán, bảo mật, người bán)"
    )

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
        "Sàn hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để đổi phương thức thanh toán đơn hàng?",
        "Quy định về đăng bán sản phẩm cho người bán?",
        "Cách mua hàng trên sàn của quốc gia khác?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{hash(s)}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)

    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.divider()
    st.caption("**Kiến trúc hệ thống:**")
    st.caption(
        "Hybrid Retrieval (Semantic + BM25) → RRF Rerank → "
        "PageIndex Fallback → LLM Generation có Citation"
    )

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("🛒 E-commerce Support RAG Chatbot")
st.caption("Hệ thống hỏi đáp chính sách e-commerce và trợ giúp khách hàng")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []), msg.get("retrieval_source"))

# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/hỗ trợ e-commerce...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Sinh câu trả lời từ RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            try:
                result = ask_rag(query, top_k=top_k)
                answer = result["answer"]
                sources = result["sources"]
                retrieval_source = result["retrieval_source"]
            except NotImplementedError:
                answer = (
                    "⚠️ **Task 10 chưa được implement.** "
                    "Hãy hoàn thành `src/task10_generation.py` để kết nối pipeline vào UI!"
                )
                sources = []
                retrieval_source = None
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                sources = []
                retrieval_source = None

            st.markdown(answer)
            render_sources(sources, retrieval_source)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
        }
    )

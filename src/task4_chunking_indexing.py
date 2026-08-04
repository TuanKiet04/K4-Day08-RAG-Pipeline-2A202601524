"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


import os
from dotenv import load_dotenv

# Load các biến môi trường (API Keys) từ file .env
load_dotenv()

# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 1000        # Data Tiki chứa văn bản dạng điều khoản, chính sách khá dài và liền mạch. Chọn 1000 ký tự để mỗi chunk giữ được 1 đoạn văn hoàn chỉnh (khoảng 200 từ).
CHUNK_OVERLAP = 200      # Chọn overlap 200 ký tự (khoảng 2-3 câu) để các đoạn cắt (chunks) không bị đứt gãy mạch ý nghĩa, đảm bảo câu văn ở chỗ giao cắt vẫn đủ ngữ cảnh.
CHUNKING_METHOD = "recursive"  # "recursive": phương pháp an toàn nhất, tự động ưu tiên cắt ở các dấu ngắt đoạn (\n\n), rồi đến dấu chấm câu (.), giúp câu văn được giữ trọn vẹn nhất có thể.

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "openai/text-embedding-3-small"  # Sử dụng model của OpenAI thông qua OpenRouter. Hỗ trợ đa ngôn ngữ (tiếng Việt) cực kỳ mạnh mẽ, chất lượng cao hơn hẳn các model local nhỏ.
EMBEDDING_DIM = 1536 # Kích thước chiều của text-embedding-3-small

# TODO: Chọn vector store
VECTOR_STORE = "chromadb"  # "chromadb": Dễ cài đặt, chạy local không cần setup Docker, tự động lưu dữ liệu xuống đĩa (persistent). Phù hợp nhất cho hệ thống nhẹ.
COLLECTION_NAME = "ecommerce_support_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy kết hợp (Hybrid: MarkdownHeader + Recursive).

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    
    # 1. Cấu hình cắt theo thẻ Heading của Markdown (để lấy ngữ cảnh từ tiêu đề)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # 2. Cấu hình cắt theo độ dài (áp dụng cho những phần nằm dưới thẻ heading mà quá dài)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        # Bước 1: Cắt nội dung theo Heading
        md_header_splits = markdown_splitter.split_text(doc["content"])
        
        # Bước 2: Dùng recursive để cắt nhỏ tiếp các đoạn nếu chúng vượt quá CHUNK_SIZE
        final_splits = recursive_splitter.split_documents(md_header_splits)
        
        for i, split in enumerate(final_splits):
            # Kết hợp metadata gốc (tên file, loại tài liệu) với metadata mới (tên Heading 1, 2, 3)
            merged_metadata = {**doc["metadata"], **split.metadata, "chunk_index": i}
            chunks.append({
                "content": split.page_content,
                "metadata": merged_metadata
            })
            
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from openai import OpenAI
    
    # Sử dụng OpenRouter API để lấy Embedding của OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    
    texts = [c["content"] for c in chunks]
    
    # OpenRouter/OpenAI API có giới hạn batch size, nhưng với số lượng chunks nhỏ (<2000), 
    # chúng ta có thể gọi 1 lần (hoặc bạn có thể chia lô nếu data quá lớn)
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    
    embeddings = [data.embedding for data in response.data]
    
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import chromadb
    
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    
    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()

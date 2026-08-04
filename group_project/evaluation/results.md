# RAG Evaluation Results

## Framework sử dụng

RAGAS 0.1.21 với OpenAI judge model `gpt-4o-mini` và embedding `text-embedding-3-small`.
Metric Answer Relevance dùng `answer_correctness reported as answer_relevance` vì benchmark có expected_answer.

- Generated at: 2026-08-04T20:39:05
- Golden dataset: D:/Users/ADMIN/LocalData/Work/Labworks/K4-Day08-RAG-Pipeline-2A202601524/group_project/evaluation/golden_dataset.json
- Number of test cases: 15
- Actual output được sinh bằng `src.task10_generation.generate_answer_from_chunks()` từ context Task 9.

## Execution Diagnostics

- Config A retrieval source distribution: {'hybrid': 15}
- Config B retrieval source distribution: {'hybrid': 15}
- Config A expected-source hit/top1/MRR: {'expected_source_hit_rate': 1.0, 'expected_source_top1_rate': 0.8, 'expected_source_mrr': 0.86}
- Config B expected-source hit/top1/MRR: {'expected_source_hit_rate': 1.0, 'expected_source_top1_rate': 0.8, 'expected_source_mrr': 0.86}
- Sanity metrics bên dưới không thay RAGAS; dùng để kiểm tra metric không bị cao giả do chỉ trúng file source.

---

## Overall Scores (RAGAS)

| Metric | Config A (hybrid + rerank) | Config B (hybrid, no rerank) | Δ |
|--------|---------------------------|------------------------------|---|
| Faithfulness | 0.9044 | 0.8643 | +0.0401 |
| Answer Relevance | 0.7895 | 0.7547 | +0.0348 |
| Context Recall | 0.9333 | 0.9333 | +0.0000 |
| Context Precision | 0.9786 | 0.9558 | +0.0228 |
| **Average** | 0.9014 | 0.8770 | +0.0244 |

## Sanity Scores (deterministic cross-check)

| Metric | Config A (hybrid + rerank) | Config B (hybrid, no rerank) | Δ |
|--------|---------------------------|------------------------------|---|
| Faithfulness | 0.9643 | 0.9604 | +0.0039 |
| Answer Relevance | 0.6779 | 0.6324 | +0.0455 |
| Context Recall | 0.9331 | 0.9331 | +0.0000 |
| Context Precision | 0.6667 | 0.6667 | +0.0000 |
| **Average** | 0.8105 | 0.7982 | +0.0123 |

---

## A/B Comparison Analysis

**Config A:**
Hybrid retrieval từ semantic search + lexical BM25, merge bằng RRF và bật reranking.

**Config B:**
Hybrid retrieval từ semantic search + lexical BM25, merge bằng RRF nhưng tắt reranking ở bước cuối.

**Kết luận:**
Config A (hybrid + rerank) có điểm RAGAS trung bình cao hơn hoặc bằng trên bộ 15 câu benchmark. Nếu RAGAS và sanity metric lệch lớn, ưu tiên đọc Worst Performers để xác định lỗi retrieval hay generation.

---

## Worst Performers (Bottom 3 - Config A)

| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|-----------|---------------|------------|
| 1 | Đơn hàng COD trên Tiki được áp dụng tối đa cho giá trị bao nhiêu? | 0.0000 | 0.0700 | 0.0000 | 0.6792 | Retrieval | Thiếu evidence trong top-k; expected source rank=5 |
| 2 | Điều kiện đổi trả khi khách hàng đổi ý trên Tiki là gì? | 0.6667 | 0.8233 | 1.0000 | 1.0000 | Context filtering | Top-k có nhiều chunk nhiễu hoặc chunk đúng nằm sâu |
| 3 | Tiki thu thập thông tin cá nhân của khách hàng cho những mục đích nào? | 0.9000 | 0.6068 | 1.0000 | 1.0000 | Minor quality gap | Điểm thấp nhẹ do khác biệt wording/citation |

## Case-level Sanity Check (Config A)

| # | Expected Source Rank | Sanity Recall | Sanity Precision | Answer Preview |
|---|----------------------|---------------|------------------|----------------|
| 1 | 2 | 1.0000 | 0.6000 | Tiki hỗ trợ các phương thức thanh toán sau: 1. Thanh toán khi nhận hàng (COD). 2. Thanh toán qua thẻ ATM có đăng ký than |
| 2 | 5 | 0.5882 | 0.6000 | Tôi không thể xác minh thông tin này từ nguồn hiện có. |
| 3 | 1 | 1.0000 | 1.0000 | Tiki hoàn tiền cho đơn hàng thanh toán tiền mặt bằng cách chuyển khoản hoặc tiki xu sau khi nhận được sản phẩm trả về củ |
| 4 | 5 | 1.0000 | 0.6000 | Thời gian hoàn tiền đối với hàng đổi trả trên Tiki sẽ được tiến hành ngay sau khi quy trình kiểm tra đánh giá chất lượng |
| 5 | 1 | 0.9655 | 0.2000 | Khách hàng được kiểm hàng trên Tiki trong phạm vi mở niêm phong thùng hàng để kiểm tra hàng hóa, nhưng không được mở sea |
| 6 | 1 | 1.0000 | 0.8000 | Chính sách bảo mật thanh toán của Tiki nêu những cơ chế bảo mật sau: 1. Thông tin tài chính của khách hàng được bảo vệ b |
| 7 | 1 | 0.7838 | 0.8000 | Tiki thu thập thông tin cá nhân của khách hàng cho các mục đích sau: 1. Cải thiện và cá nhân hóa trải nghiệm mua sắm trê |
| 8 | 1 | 1.0000 | 0.6000 | Khách hàng có thể gửi khiếu nại tới Tiki qua các kênh sau: gọi điện thoại đến hotline 19006035, gửi thư điện tử đến địa  |
| 9 | 1 | 0.9714 | 1.0000 | Khi giải quyết khiếu nại, Tiki có thể yêu cầu Khách hàng và/hoặc Nhà Bán Hàng cung cấp các thông tin, bằng chứng liên qu |
| 10 | 1 | 0.9737 | 0.6000 | Thời gian hỗ trợ đổi trả sản phẩm mua trên Tiki là trong vòng 30 ngày kể từ lúc nhận hàng thành công, ngoại trừ một số s |
| 11 | 1 | 1.0000 | 0.2000 | Điều kiện đổi trả khi khách hàng đổi ý trên Tiki bao gồm: 1. Sản phẩm còn nguyên tình trạng như khi nhận, hộp và bao bì  |
| 12 | 1 | 0.7619 | 0.6000 | Nếu sản phẩm giao đến bị hư hỏng, không đảm bảo chất lượng, giao sai sản phẩm hoặc sai số lượng, khách hàng cần đồng kiể |
| 13 | 1 | 1.0000 | 0.8000 | Tiki có các hình thức giao hàng sau: 1. **Lưu kho Tiki (FBT)**: Nhà Bán Hàng gửi Hàng Hóa vào Kho Tiki, Tiki sẽ xử lý to |
| 14 | 1 | 0.9524 | 1.0000 | Điểm TikiVIP có giá trị quy đổi là mỗi 1 Điểm TikiVIP tương đương 1 VNĐ và chỉ được sử dụng tối đa 50% giá trị của mỗi đ |
| 15 | 1 | 1.0000 | 0.6000 | Điều kiện để lên hạng Vàng trong chương trình Tiki VIP là bạn cần đạt ít nhất 3 đơn hàng và chi tiêu từ 1,000,000 VND tr |

---

## Recommendations

### Cải tiến 1
**Action:** So sánh thêm Config C dense-only hoặc lexical-only để A/B có khác biệt rõ hơn RRF-on/off.
**Expected impact:** Xác định chính xác phần đóng góp của semantic search, BM25 và reranking.

### Cải tiến 2
**Action:** Bổ sung metadata section/source rõ hơn khi chunking và ưu tiên section match khi rerank.
**Expected impact:** Tăng Context Precision cho câu hỏi bám vào mục chính sách cụ thể.

### Cải tiến 3
**Action:** Tinh chỉnh prompt Task 10 theo hướng trả lời ngắn, đủ ý, tránh kéo thêm context không liên quan.
**Expected impact:** Tăng Answer Relevance và giảm nguy cơ câu trả lời lan man.

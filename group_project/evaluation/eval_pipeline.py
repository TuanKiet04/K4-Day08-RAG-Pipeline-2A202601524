"""
RAG Evaluation Pipeline.

Yêu cầu repo:
    1. Load golden_dataset.json (>=15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, answer_relevance,
       context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Triển khai hiện tại:
    - Dùng Task 9 để retrieve theo từng config A/B.
    - Dùng Task 10 để sinh actual answer có citation từ retrieved chunks.
    - Dùng RAGAS để chấm 4 metrics chính theo yêu cầu README.
    - Tính thêm deterministic sanity metrics để phát hiện kết quả "đẹp giả"
      như chỉ cần trúng file source là recall/precision = 1.0.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv


EVALUATION_DIR = Path(__file__).parent
PROJECT_ROOT = EVALUATION_DIR.parents[1]
GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
RESULTS_PATH = EVALUATION_DIR / "results.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


REQUIRED_FIELDS = {"question", "expected_answer", "expected_context"}
METRIC_KEYS = ("faithfulness", "answer_relevance", "context_recall", "context_precision")
RAGAS_TO_LOCAL_KEYS = {
    "faithfulness": "faithfulness",
    # Với bộ benchmark tiếng Việt có expected_answer, answer_correctness ổn định hơn
    # answer_relevancy của RAGAS 0.1.x (metric này có thể trả 0 dù answer đúng).
    "answer_correctness": "answer_relevance",
    "context_recall": "context_recall",
    "context_precision": "context_precision",
}

JUDGE_MODEL = os.getenv("EVAL_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
if JUDGE_MODEL.startswith("openai/"):
    JUDGE_MODEL = JUDGE_MODEL.removeprefix("openai/")


@dataclass(frozen=True)
class EvalConfig:
    """Một cấu hình retrieval để so sánh A/B."""

    name: str
    label: str
    description: str
    use_reranking: bool
    score_threshold: float = 0.3
    top_k: int = 5


CONFIGS = [
    EvalConfig(
        name="hybrid_rerank",
        label="Config A (hybrid + rerank)",
        description="Hybrid retrieval từ semantic search + lexical BM25, merge bằng RRF và bật reranking.",
        use_reranking=True,
    ),
    EvalConfig(
        name="hybrid_no_rerank",
        label="Config B (hybrid, no rerank)",
        description="Hybrid retrieval từ semantic search + lexical BM25, merge bằng RRF nhưng tắt reranking ở bước cuối.",
        use_reranking=False,
    ),
]


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "và",
    "của",
    "có",
    "cho",
    "các",
    "là",
    "khi",
    "với",
    "được",
    "trong",
    "trên",
    "về",
    "để",
    "từ",
    "này",
    "những",
    "một",
    "hoặc",
    "nếu",
    "thì",
    "tại",
    "theo",
    "khách",
    "hàng",
    "tiki",
}


def load_golden_dataset() -> list[dict]:
    """Load và validate golden dataset."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("golden_dataset.json phải là JSON array")
    if len(data) < 15:
        raise ValueError("golden_dataset.json phải có tối thiểu 15 Q&A pairs")

    for idx, item in enumerate(data, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Case #{idx} phải là object")
        missing = REQUIRED_FIELDS - set(item)
        if missing:
            raise ValueError(f"Case #{idx} thiếu field: {sorted(missing)}")
        for field in REQUIRED_FIELDS:
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f"Case #{idx} field {field!r} phải là non-empty string")

    return data


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản, giữ Unicode tiếng Việt."""
    return re.findall(r"[0-9A-Za-zÀ-ỹ]+", (text or "").lower())


def _meaningful_tokens(text: str) -> set[str]:
    """Token set sau khi bỏ stopwords rất phổ biến để sanity metric bớt ảo."""
    return {t for t in _tokenize(text) if len(t) > 1 and t not in STOPWORDS}


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _token_recall(reference: str, candidate: str) -> float:
    ref_tokens = _meaningful_tokens(reference)
    cand_tokens = _meaningful_tokens(candidate)
    return _safe_div(len(ref_tokens & cand_tokens), len(ref_tokens))


def _token_precision(candidate: str, reference: str) -> float:
    cand_tokens = _meaningful_tokens(candidate)
    ref_tokens = _meaningful_tokens(reference)
    return _safe_div(len(cand_tokens & ref_tokens), len(cand_tokens))


def _token_f1(a: str, b: str) -> float:
    precision = _token_precision(a, b)
    recall = _token_recall(b, a)
    return _safe_div(2 * precision * recall, precision + recall)


def _strip_citations(text: str) -> str:
    return re.sub(r"\[[^\]]+\]", " ", text or "")


def _expected_source_name(expected_context: str) -> str:
    """Rút tên file source từ expected_context nếu có."""
    match = re.search(r"([A-Za-z0-9_.-]+\.md)", expected_context or "")
    return match.group(1) if match else ""


def _source_names(sources: list[dict]) -> list[str]:
    names = []
    for source in sources:
        metadata = source.get("metadata") or {}
        source_name = metadata.get("source")
        if source_name:
            names.append(str(source_name))
    return names


def _expected_source_rank(case: dict) -> int | None:
    expected_source = _expected_source_name(case["expected_context"])
    if not expected_source:
        return None
    for idx, source_name in enumerate(_source_names(case["sources"]), 1):
        if source_name == expected_source:
            return idx
    return None


def _context_text(sources: list[dict]) -> str:
    return "\n".join(str(source.get("content", "")) for source in sources)


def _average(metrics: dict[str, float]) -> float:
    return round(mean(metrics[key] for key in METRIC_KEYS), 4)


def _deterministic_sanity_metrics(item: dict, actual_answer: str, sources: list[dict]) -> dict[str, float]:
    """
    Sanity metrics offline, không thay RAGAS.

    Mục tiêu là kiểm tra xu hướng hợp lý:
    - Không cho context recall = 1 chỉ vì retrieve trúng đúng file.
    - Context precision chỉ tính chunk hữu ích nếu chunk chứa đủ evidence đáng kể.
    """
    expected_answer = item["expected_answer"]
    all_context = _context_text(sources)
    answer_no_cites = _strip_citations(actual_answer)

    answer_relevance = _token_f1(answer_no_cites, expected_answer)
    faithfulness = _token_precision(answer_no_cites, all_context) if all_context else 0.0
    context_recall = _token_recall(expected_answer, all_context)

    if not sources:
        context_precision = 0.0
    else:
        useful = 0
        for source in sources:
            content = str(source.get("content", ""))
            evidence_recall = _token_recall(expected_answer, content)
            evidence_f1 = _token_f1(content, expected_answer)
            if evidence_recall >= 0.35 or evidence_f1 >= 0.22:
                useful += 1
        context_precision = useful / len(sources)

    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevance": round(answer_relevance, 4),
        "context_recall": round(context_recall, 4),
        "context_precision": round(context_precision, 4),
    }


def _aggregate_cases(cases: list[dict], metric_field: str = "metrics") -> dict[str, float]:
    if not cases:
        return {key: 0.0 for key in METRIC_KEYS} | {"average": 0.0}

    scores = {
        key: round(mean(float(case[metric_field][key]) for case in cases), 4)
        for key in METRIC_KEYS
    }
    scores["average"] = _average(scores)
    return scores


def _source_distribution(cases: list[dict]) -> dict[str, int]:
    """Đếm top-level retrieval source để phát hiện fallback khi chạy eval."""
    return dict(Counter(case.get("retrieval_source", "none") for case in cases))


def _source_rank_stats(cases: list[dict]) -> dict[str, float]:
    ranks = [_expected_source_rank(case) for case in cases]
    hits = [rank for rank in ranks if rank is not None]
    top1 = sum(1 for rank in ranks if rank == 1)
    return {
        "expected_source_hit_rate": round(len(hits) / len(cases), 4) if cases else 0.0,
        "expected_source_top1_rate": round(top1 / len(cases), 4) if cases else 0.0,
        "expected_source_mrr": round(mean(1 / rank for rank in hits), 4) if hits else 0.0,
    }


def run_pipeline_for_case(item: dict, config: EvalConfig) -> dict:
    """Chạy retrieval theo config rồi sinh answer bằng Task 10."""
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import generate_answer_from_chunks

    sources = retrieve(
        item["question"],
        top_k=config.top_k,
        score_threshold=config.score_threshold,
        use_reranking=config.use_reranking,
    )
    generation = generate_answer_from_chunks(item["question"], sources)
    return {
        "answer": generation["answer"],
        "sources": generation["sources"],
        "retrieval_source": generation.get(
            "retrieval_source",
            sources[0].get("source", "none") if sources else "none",
        ),
    }


def _collect_cases(config: EvalConfig, golden_dataset: list[dict]) -> list[dict]:
    cases = []
    for idx, item in enumerate(golden_dataset, 1):
        print(f"[{config.name}] generate {idx}/{len(golden_dataset)} {item['question']}")
        result = run_pipeline_for_case(item, config)
        sanity_metrics = _deterministic_sanity_metrics(item, result["answer"], result["sources"])
        cases.append(
            {
                "id": idx,
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "expected_context": item["expected_context"],
                "actual_answer": result["answer"],
                "sources": result["sources"],
                "retrieval_source": result["retrieval_source"],
                "sanity_metrics": sanity_metrics,
                "sanity_average": _average(sanity_metrics),
                "metrics": {key: 0.0 for key in METRIC_KEYS},
                "average": 0.0,
            }
        )
    return cases


def _score_with_ragas(cases: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Chấm cases bằng RAGAS và map score về schema nội bộ."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Thiếu OPENAI_API_KEY để chạy RAGAS judge")

    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_correctness,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig
    from src.task4_chunking_indexing import EMBEDDING_MODEL

    dataset = Dataset.from_dict(
        {
            "question": [case["question"] for case in cases],
            "answer": [case["actual_answer"] for case in cases],
            "contexts": [[str(src.get("content", "")) for src in case["sources"]] for case in cases],
            "ground_truth": [case["expected_answer"] for case in cases],
        }
    )

    llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    result = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_correctness, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=180, max_retries=3, max_wait=30, max_workers=4),
        raise_exceptions=False,
    )
    df = result.to_pandas()

    scored_cases = []
    for case, (_, row) in zip(cases, df.iterrows()):
        metrics: dict[str, float] = {}
        for ragas_key, local_key in RAGAS_TO_LOCAL_KEYS.items():
            value = row.get(ragas_key, 0.0)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = 0.0
            if numeric != numeric:  # NaN
                numeric = 0.0
            metrics[local_key] = round(max(0.0, min(1.0, numeric)), 4)
        scored = case.copy()
        scored["metrics"] = metrics
        scored["average"] = _average(metrics)
        scored_cases.append(scored)

    metadata = {
        "framework": "RAGAS",
        "framework_version": _get_package_version("ragas"),
        "judge_model": JUDGE_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "answer_metric": "answer_correctness reported as answer_relevance",
    }
    return scored_cases, metadata


def _get_package_version(package: str) -> str:
    try:
        import importlib.metadata as metadata

        return metadata.version(package)
    except Exception:
        return "unknown"


def evaluate_config(config: EvalConfig, golden_dataset: list[dict]) -> dict:
    """Evaluate toàn bộ golden dataset với một config."""
    cases = _collect_cases(config, golden_dataset)
    print(f"[{config.name}] score with RAGAS")
    cases, framework_metadata = _score_with_ragas(cases)
    return {
        "config": config,
        "framework_metadata": framework_metadata,
        "scores": _aggregate_cases(cases, "metrics"),
        "sanity_scores": _aggregate_cases(cases, "sanity_metrics"),
        "source_distribution": _source_distribution(cases),
        "source_rank_stats": _source_rank_stats(cases),
        "cases": cases,
    }


def compare_configs(rag_pipeline: Any, golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa 2 configs.

    Tham số rag_pipeline giữ lại để tương thích skeleton ban đầu.
    """
    _ = rag_pipeline
    return {config.name: evaluate_config(config, golden_dataset) for config in CONFIGS}


def _bottom_cases(config_result: dict, limit: int = 3) -> list[dict]:
    return sorted(config_result["cases"], key=lambda case: case["average"])[:limit]


def _failure_stage(case: dict) -> str:
    metrics = case["metrics"]
    sanity = case["sanity_metrics"]
    if metrics["context_recall"] < 0.5 or sanity["context_recall"] < 0.35:
        return "Retrieval"
    if metrics["context_precision"] < 0.5 or sanity["context_precision"] < 0.35:
        return "Context filtering"
    if metrics["faithfulness"] < 0.5:
        return "Grounding"
    if metrics["answer_relevance"] < 0.5:
        return "Answer generation"
    return "Minor quality gap"


def _root_cause(case: dict) -> str:
    metrics = case["metrics"]
    sanity = case["sanity_metrics"]
    expected_rank = _expected_source_rank(case)
    if not case["sources"]:
        return "Không retrieve được context"
    if expected_rank is None:
        return "Không thấy expected source trong top-k"
    if metrics["context_recall"] < 0.5 or sanity["context_recall"] < 0.35:
        return f"Thiếu evidence trong top-k; expected source rank={expected_rank}"
    if metrics["context_precision"] < 0.5 or sanity["context_precision"] < 0.35:
        return "Top-k có nhiều chunk nhiễu hoặc chunk đúng nằm sâu"
    if metrics["answer_relevance"] < 0.5:
        return "LLM trả lời chưa khớp đầy đủ expected answer"
    return "Điểm thấp nhẹ do khác biệt wording/citation"


def _metric_delta(a: float, b: float) -> str:
    delta = round(a - b, 4)
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def _markdown_metric_table(scores_a: dict, scores_b: dict) -> list[str]:
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
        "average": "**Average**",
    }
    lines = [
        "| Metric | Config A (hybrid + rerank) | Config B (hybrid, no rerank) | Δ |",
        "|--------|---------------------------|------------------------------|---|",
    ]
    for key in (*METRIC_KEYS, "average"):
        lines.append(
            f"| {metric_labels[key]} | {scores_a[key]:.4f} | {scores_b[key]:.4f} | "
            f"{_metric_delta(scores_a[key], scores_b[key])} |"
        )
    return lines


def export_results(results: dict, comparison: dict | None = None):
    """Export evaluation results to results.md."""
    _ = results
    comparison = comparison or results

    config_a = comparison["hybrid_rerank"]
    config_b = comparison["hybrid_no_rerank"]
    scores_a = config_a["scores"]
    scores_b = config_b["scores"]
    sanity_a = config_a["sanity_scores"]
    sanity_b = config_b["sanity_scores"]
    dist_a = config_a["source_distribution"]
    dist_b = config_b["source_distribution"]
    rank_a = config_a["source_rank_stats"]
    rank_b = config_b["source_rank_stats"]
    framework = config_a["framework_metadata"]

    better = (
        config_a["config"].label
        if scores_a["average"] >= scores_b["average"]
        else config_b["config"].label
    )

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Framework sử dụng",
        "",
        f"RAGAS {framework['framework_version']} với OpenAI judge model `{framework['judge_model']}` "
        f"và embedding `{framework['embedding_model']}`.",
        f"Metric Answer Relevance dùng `{framework['answer_metric']}` vì benchmark có expected_answer.",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Golden dataset: {GOLDEN_DATASET_PATH.as_posix()}",
        f"- Number of test cases: {len(config_a['cases'])}",
        "- Actual output được sinh bằng `src.task10_generation.generate_answer_from_chunks()` từ context Task 9.",
        "",
        "## Execution Diagnostics",
        "",
        f"- Config A retrieval source distribution: {dist_a}",
        f"- Config B retrieval source distribution: {dist_b}",
        f"- Config A expected-source hit/top1/MRR: {rank_a}",
        f"- Config B expected-source hit/top1/MRR: {rank_b}",
        "- Sanity metrics bên dưới không thay RAGAS; dùng để kiểm tra metric không bị cao giả do chỉ trúng file source.",
        "",
        "---",
        "",
        "## Overall Scores (RAGAS)",
        "",
    ]
    lines.extend(_markdown_metric_table(scores_a, scores_b))

    lines.extend(
        [
            "",
            "## Sanity Scores (deterministic cross-check)",
            "",
        ]
    )
    lines.extend(_markdown_metric_table(sanity_a, sanity_b))

    lines.extend(
        [
            "",
            "---",
            "",
            "## A/B Comparison Analysis",
            "",
            "**Config A:**",
            config_a["config"].description,
            "",
            "**Config B:**",
            config_b["config"].description,
            "",
            "**Kết luận:**",
            f"{better} có điểm RAGAS trung bình cao hơn hoặc bằng trên bộ 15 câu benchmark. "
            "Nếu RAGAS và sanity metric lệch lớn, ưu tiên đọc Worst Performers để xác định lỗi retrieval hay generation.",
            "",
            "---",
            "",
            "## Worst Performers (Bottom 3 - Config A)",
            "",
            "| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |",
            "|---|----------|-------------|-----------|--------|-----------|---------------|------------|",
        ]
    )

    for idx, case in enumerate(_bottom_cases(config_a), 1):
        metrics = case["metrics"]
        question = case["question"].replace("|", "\\|")
        lines.append(
            f"| {idx} | {question} | {metrics['faithfulness']:.4f} | "
            f"{metrics['answer_relevance']:.4f} | {metrics['context_recall']:.4f} | "
            f"{metrics['context_precision']:.4f} | {_failure_stage(case)} | {_root_cause(case)} |"
        )

    lines.extend(
        [
            "",
            "## Case-level Sanity Check (Config A)",
            "",
            "| # | Expected Source Rank | Sanity Recall | Sanity Precision | Answer Preview |",
            "|---|----------------------|---------------|------------------|----------------|",
        ]
    )
    for case in config_a["cases"]:
        answer_preview = " ".join(case["actual_answer"].split())[:120].replace("|", "\\|")
        rank = _expected_source_rank(case)
        rank_text = str(rank) if rank is not None else "miss"
        sanity = case["sanity_metrics"]
        lines.append(
            f"| {case['id']} | {rank_text} | {sanity['context_recall']:.4f} | "
            f"{sanity['context_precision']:.4f} | {answer_preview} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Recommendations",
            "",
            "### Cải tiến 1",
            "**Action:** So sánh thêm Config C dense-only hoặc lexical-only để A/B có khác biệt rõ hơn RRF-on/off.",
            "**Expected impact:** Xác định chính xác phần đóng góp của semantic search, BM25 và reranking.",
            "",
            "### Cải tiến 2",
            "**Action:** Bổ sung metadata section/source rõ hơn khi chunking và ưu tiên section match khi rerank.",
            "**Expected impact:** Tăng Context Precision cho câu hỏi bám vào mục chính sách cụ thể.",
            "",
            "### Cải tiến 3",
            "**Action:** Tinh chỉnh prompt Task 10 theo hướng trả lời ngắn, đủ ý, tránh kéo thêm context không liên quan.",
            "**Expected impact:** Tăng Answer Relevance và giảm nguy cơ câu trả lời lan man.",
            "",
        ]
    )

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")
    comparison = compare_configs(None, golden_dataset)
    export_results(comparison, comparison)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run golden queries against the RAG pipeline and report results.

Usage:
    python scripts/evaluate.py                  # Run all queries
    python scripts/evaluate.py --query q001     # Run single query
    python scripts/evaluate.py --save-baseline  # Save results as baseline
    python scripts/evaluate.py --compare        # Compare against baseline
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import yaml

# Ensure project root is on the path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def load_golden_queries(path: str = None) -> list:
    """Load golden queries from YAML file."""
    if path is None:
        path = str(PROJECT_ROOT / "data" / "evaluation" / "golden_queries.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("queries", [])


def check_doc_recall(response, expected_docs: list) -> float:
    """Check what fraction of expected documents appear in citations."""
    if not expected_docs:
        return 1.0  # No expectation = automatic pass.

    citations = response.citations or []
    cited_texts = " ".join(
        str(c.get("document_title", "") or c.get("filename", ""))
        for c in citations
    ).lower()

    # Also check the answer text for document references.
    answer_lower = (response.answer or "").lower()
    combined = cited_texts + " " + answer_lower

    found = sum(1 for doc in expected_docs if doc.lower() in combined)
    return found / len(expected_docs)


def check_concept_coverage(response, required: list, forbidden: list) -> float:
    """Check required concepts are present and forbidden are absent."""
    answer_lower = (response.answer or "").lower()

    if forbidden:
        for concept in forbidden:
            if concept.lower() in answer_lower:
                return 0.0  # Fail if forbidden concept found.

    if not required:
        return 1.0

    found = sum(1 for c in required if c.lower() in answer_lower)
    return found / len(required)


def check_citations(response, min_citations: int) -> bool:
    """Check minimum citation count."""
    actual = len(response.citations or [])
    return actual >= min_citations


async def run_query(pipeline, query_spec: dict) -> dict:
    """Run a single golden query and return evaluation result."""
    query = query_spec["query"]
    query_id = query_spec["id"]

    start = time.perf_counter()
    try:
        response = await pipeline.process_query(query)
        elapsed_ms = (time.perf_counter() - start) * 1000
    except Exception as e:
        return {
            "id": query_id,
            "passed": False,
            "error": str(e),
            "elapsed_ms": 0,
        }

    recall = check_doc_recall(response, query_spec.get("expected_docs", []))
    concepts = check_concept_coverage(
        response,
        query_spec.get("required_concepts", []),
        query_spec.get("forbidden_concepts", []),
    )
    cit_ok = check_citations(response, query_spec.get("min_citations", 0))

    passed = recall >= 0.5 and concepts >= 0.5 and cit_ok

    return {
        "id": query_id,
        "passed": passed,
        "recall": round(recall, 2),
        "concepts": round(concepts, 2),
        "citations_ok": cit_ok,
        "confidence": round(response.confidence, 2),
        "elapsed_ms": round(elapsed_ms, 1),
        "failure_reason": (
            None if passed
            else (
                "retrieval: expected doc not found" if recall < 0.5
                else "concepts: missing required concepts" if concepts < 0.5
                else "citations: below minimum"
            )
        ),
    }


def print_results(results: list) -> None:
    """Print a pass/fail table."""
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    pct = (passed / total * 100) if total else 0

    print(f"\nAAIRE Eval -- {passed}/{total} passed ({pct:.0f}%)\n")
    print(f"{'ID':<8} {'Status':<6} {'Recall':<10} {'Concepts':<12} {'Time':<10} {'Reason'}")
    print("-" * 70)

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        recall = f"{r.get('recall', 0):.2f}" if "recall" in r else "N/A"
        concepts = f"{r.get('concepts', 0):.0%}" if "concepts" in r else "N/A"
        time_str = f"{r.get('elapsed_ms', 0):.0f}ms"
        reason = r.get("failure_reason") or r.get("error", "")
        print(f"{r['id']:<8} {status:<6} {recall:<10} {concepts:<12} {time_str:<10} {reason}")


def save_baseline(results: list, path: str = None) -> None:
    """Save results as baseline JSON."""
    if path is None:
        path = str(PROJECT_ROOT / "data" / "evaluation" / "baseline.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline saved to {path}")


def compare_baseline(results: list, path: str = None) -> None:
    """Compare current results against saved baseline."""
    if path is None:
        path = str(PROJECT_ROOT / "data" / "evaluation" / "baseline.json")
    if not os.path.exists(path):
        print("No baseline found. Run with --save-baseline first.")
        return

    with open(path) as f:
        baseline = json.load(f)

    baseline_map = {r["id"]: r for r in baseline}
    regressions = []

    for r in results:
        b = baseline_map.get(r["id"])
        if b and b["passed"] and not r["passed"]:
            regressions.append(r["id"])

    if regressions:
        print(f"\nREGRESSIONS detected: {', '.join(regressions)}")
    else:
        print("\nNo regressions vs baseline.")


async def main():
    parser = argparse.ArgumentParser(description="AAIRE RAG evaluation harness")
    parser.add_argument("--query", type=str, help="Run a single query by ID (e.g., q001)")
    parser.add_argument("--save-baseline", action="store_true", help="Save results as baseline")
    parser.add_argument("--compare", action="store_true", help="Compare against saved baseline")
    parser.add_argument("--golden-file", type=str, help="Path to golden queries YAML")
    args = parser.parse_args()

    # Load golden queries.
    queries = load_golden_queries(args.golden_file)

    if args.query:
        queries = [q for q in queries if q["id"] == args.query]
        if not queries:
            print(f"Query ID '{args.query}' not found in golden set.")
            sys.exit(1)

    # Initialize the RAG pipeline.
    print("Initializing RAG pipeline...")
    from src.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()

    # Run evaluations.
    print(f"Running {len(queries)} golden queries...\n")
    results = []
    for spec in queries:
        result = await run_query(pipeline, spec)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {result['id']}: {status}")
        results.append(result)

    # Output.
    print_results(results)

    if args.save_baseline:
        save_baseline(results)

    if args.compare:
        compare_baseline(results)

    # Exit code: non-zero if any failures.
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    asyncio.run(main())

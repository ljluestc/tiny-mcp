#!/usr/bin/env python3
"""DevOps & LLM Interview Knowledge MCP Service.

Serves interview Q&A from RAG datasets as MCP tools.
Loads JSON datasets and provides search/browse/quiz capabilities.

Datasets:
  - devops_rag_answered_full_405_regenerated.json  (DevOps: 405 Q&A)
  - llm_interview_note_qa.json                     (LLM: 405 Q&A)

Usage:
    python services/devops_knowledge_service.py
    # Then connect via anthropic_mcp_client.py
"""

import json
import random
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DevOpsKnowledge")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
RAG_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_datasets: dict[str, list[dict]] = {}


def _load_datasets():
    """Load JSON datasets lazily on first use."""
    if _datasets:
        return

    # Try multiple locations for the data files
    search_dirs = [
        RAG_DATA_DIR,
        Path.home() / "dev" / "RAG" / "kubernetes_rag" / "data" / "devops_rag_kb",
        Path.home() / "dev" / "devops-interview-questions" / "data",
    ]

    devops_files = [
        "devops_rag_answered_full_405_regenerated.json",
        "devops_rag_answered_full_405.json",
    ]
    llm_files = [
        "llm_interview_note_qa.json",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for fname in devops_files:
            p = search_dir / fname
            if p.exists() and "devops" not in _datasets:
                with open(p, encoding="utf-8") as f:
                    _datasets["devops"] = json.load(f)
                break
        for fname in llm_files:
            p = search_dir / fname
            if p.exists() and "llm" not in _datasets:
                with open(p, encoding="utf-8") as f:
                    _datasets["llm"] = json.load(f)
                break

    # If still not found, create a bundled fallback directory
    if not _datasets:
        RAG_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_all_entries(domain: Optional[str] = None) -> list[dict]:
    """Get entries, optionally filtered by domain (devops/llm/all)."""
    _load_datasets()
    if domain and domain in _datasets:
        return _datasets[domain]
    entries = []
    for ds in _datasets.values():
        entries.extend(ds)
    return entries


def _get_categories(domain: Optional[str] = None) -> dict[str, int]:
    """Get category counts."""
    cats: dict[str, int] = {}
    for e in _get_all_entries(domain):
        c = e.get("category", "unknown")
        cats[c] = cats.get(c, 0) + 1
    return dict(sorted(cats.items(), key=lambda x: -x[1]))


def _format_qa(entry: dict, detailed: bool = False) -> str:
    """Format a Q&A entry for display."""
    q = entry.get("question_zh") or entry.get("question", "")
    cat = entry.get("category", "")
    eid = entry.get("id", "")

    parts = [f"[{eid}] ({cat})"]
    parts.append(f"Q: {q}")

    if detailed:
        short = entry.get("short_answer_zh") or entry.get("answer", "")
        detail = entry.get("detailed_answer_zh", "")
        kp = entry.get("key_points_zh", [])

        if short:
            parts.append(f"\nShort Answer: {short}")
        if detail:
            parts.append(f"\nDetailed Answer: {detail[:1500]}")
        if kp:
            parts.append("\nKey Points:")
            for p in kp:
                parts.append(f"  - {p}")
    else:
        short = entry.get("short_answer_zh") or entry.get("answer", "")
        if short:
            parts.append(f"A: {short[:300]}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_categories(domain: Optional[str] = None) -> str:
    """List available Q&A categories with counts.

    Args:
        domain: Filter by domain - 'devops', 'llm', or None for all.
    """
    cats = _get_categories(domain)
    total = sum(cats.values())
    lines = [f"Total: {total} questions across {len(cats)} categories\n"]
    for cat, count in cats.items():
        lines.append(f"  {cat:30s} {count:4d}")
    return "\n".join(lines)


@mcp.tool()
async def search_questions(
    query: str,
    domain: Optional[str] = None,
    max_results: int = 5,
) -> str:
    """Search interview questions by keyword.

    Args:
        query: Search term (matches question text, category, keywords).
        domain: Filter by 'devops', 'llm', or None for all.
        max_results: Maximum results to return (default 5).
    """
    query_lower = query.lower()
    entries = _get_all_entries(domain)

    scored = []
    for e in entries:
        q = (e.get("question_zh") or e.get("question", "")).lower()
        cat = e.get("category", "").lower()
        kw = " ".join(e.get("keywords", [])).lower()
        answer = (e.get("short_answer_zh") or e.get("answer", "")).lower()

        score = 0
        if query_lower in q:
            score += 10
        if query_lower in cat:
            score += 5
        if query_lower in kw:
            score += 3
        if query_lower in answer:
            score += 1

        if score > 0:
            scored.append((score, e))

    scored.sort(key=lambda x: -x[0])
    results = scored[:max_results]

    if not results:
        return f"No results for '{query}'. Try broader terms or list_categories() first."

    lines = [f"Found {len(scored)} matches for '{query}' (showing top {len(results)}):\n"]
    for _, entry in results:
        lines.append(_format_qa(entry, detailed=False))
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
async def get_question_detail(question_id: str) -> str:
    """Get full detailed answer for a specific question by ID.

    Args:
        question_id: The question ID (e.g., 'devops_001', 'llm_0042').
    """
    for entries in _datasets.values():
        for e in entries:
            if e.get("id") == question_id:
                return _format_qa(e, detailed=True)
    return f"Question '{question_id}' not found."


@mcp.tool()
async def get_questions_by_category(
    category: str,
    domain: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Get questions from a specific category.

    Args:
        category: Category name (e.g., 'kubernetes', 'llm_architecture', 'jenkins').
        domain: Filter by 'devops', 'llm', or None for all.
        max_results: Maximum results (default 10).
    """
    entries = _get_all_entries(domain)
    cat_lower = category.lower()
    matches = [e for e in entries if cat_lower in e.get("category", "").lower()]

    if not matches:
        cats = _get_categories(domain)
        return f"Category '{category}' not found. Available: {', '.join(cats.keys())}"

    lines = [f"Category '{category}': {len(matches)} questions (showing {min(len(matches), max_results)}):\n"]
    for e in matches[:max_results]:
        lines.append(_format_qa(e, detailed=False))
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
async def quiz_me(
    category: Optional[str] = None,
    domain: Optional[str] = None,
    count: int = 3,
) -> str:
    """Generate a random quiz from the knowledge base.

    Args:
        category: Optional category filter.
        domain: Filter by 'devops', 'llm', or None for all.
        count: Number of questions (default 3).
    """
    entries = _get_all_entries(domain)
    if category:
        cat_lower = category.lower()
        entries = [e for e in entries if cat_lower in e.get("category", "").lower()]

    if not entries:
        return "No questions available for the given filters."

    selected = random.sample(entries, min(count, len(entries)))
    lines = [f"Quiz ({len(selected)} questions):\n"]
    for i, e in enumerate(selected, 1):
        q = e.get("question_zh") or e.get("question", "")
        eid = e.get("id", "")
        cat = e.get("category", "")
        lines.append(f"{i}. [{eid}] ({cat}) {q}")
    lines.append(f"\nUse get_question_detail(id) to see the answer for each question.")
    return "\n".join(lines)


@mcp.tool()
async def dataset_stats() -> str:
    """Get statistics about loaded datasets."""
    _load_datasets()
    lines = ["Dataset Statistics:\n"]
    total = 0
    for name, entries in _datasets.items():
        cats = {}
        for e in entries:
            c = e.get("category", "unknown")
            cats[c] = cats.get(c, 0) + 1
        lines.append(f"  {name}: {len(entries)} entries, {len(cats)} categories")
        total += len(entries)
    lines.append(f"\n  Total: {total} Q&A pairs")

    all_cats = _get_categories()
    lines.append(f"\nAll categories ({len(all_cats)}):")
    for cat, count in all_cats.items():
        lines.append(f"  {cat:30s} {count:4d}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")

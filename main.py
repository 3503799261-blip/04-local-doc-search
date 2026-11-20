"""Offline Markdown retrieval with exact excerpts and file provenance; no LLM."""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
SEED = 42


def chunks_from_directory(directory, max_chars, overlap):
    chunks = []
    for path in sorted(directory.rglob("*.md")):
        text = path.read_text(encoding="utf-8-sig")
        # Split at Markdown headings first, then use overlapping character windows.
        sections = [s.strip() for s in re.split(r"(?m)(?=^#{1,6}\s)", text) if s.strip()]
        # Keep a document title/preamble with its first substantive section,
        # so a short title alone does not outrank every useful passage.
        if len(sections) > 1 and re.match(r"^#\s", sections[0]):
            sections[1] = sections[0] + "\n\n" + sections[1]
            sections = sections[1:]
        index = 0
        for section in sections:
            section = section.strip()
            if not section:
                continue
            for start in range(0, len(section), max_chars - overlap):
                excerpt = section[start:start + max_chars]
                if not excerpt.strip():
                    continue
                index += 1
                source = path.relative_to(directory).as_posix()
                chunks.append({"id": f"{source}#{index}", "source": source, "text": excerpt})
                if start + max_chars >= len(section):
                    break
    return chunks


def keywords(text):
    clean = re.sub(r"\s+", "", text.lower())
    return {clean[i:i + 2] for i in range(max(0, len(clean) - 1))}


def rank_sources(scores, chunks):
    seen, ranking = set(), []
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] <= 0:
            continue
        source = chunks[int(i)]["source"]
        if source not in seen:
            seen.add(source)
            ranking.append(source)
    return ranking


def evaluate(scorer, queries, chunks):
    reciprocal, recalls, details = [], [], []
    for row in queries:
        ranking = rank_sources(scorer(row["query"]), chunks)
        relevant = set(row["relevant_sources"])
        positions = [rank + 1 for rank, source in enumerate(ranking) if source in relevant]
        reciprocal.append(1 / min(positions) if positions else 0.0)
        recalls.append(len(set(ranking[:3]) & relevant) / len(relevant))
        details.append({"query": row["query"], "relevant_sources": row["relevant_sources"], "top_sources": ranking[:3]})
    return {"source_mrr": float(np.mean(reciprocal)), "source_recall_at_3": float(np.mean(recalls))}, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="会议室预约后需要取消怎么办？")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "data" / "docs")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=240)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if not args.query.strip() or args.top_k < 1:
        parser.error("Provide nonempty --query and positive --top-k")
    if args.max_chars < 20 or not 0 <= args.overlap < args.max_chars:
        parser.error("Use --max-chars >= 20 and 0 <= --overlap < --max-chars")
    if not args.docs_dir.is_dir():
        parser.error("--docs-dir must be an existing directory")
    chunks = chunks_from_directory(args.docs_dir, args.max_chars, args.overlap)
    if not chunks:
        parser.error("No nonempty UTF-8 Markdown documents found")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    try:
        matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    except ValueError as exc:
        parser.error(f"Document text is too short to index: {exc}")

    def score(query):
        return (matrix @ vectorizer.transform([query]).T).toarray().ravel()

    chunk_terms = [keywords(chunk["text"]) for chunk in chunks]

    def overlap_score(query):
        query_terms = keywords(query)
        return np.array([len(query_terms & terms) for terms in chunk_terms], dtype=float)

    is_packaged = args.docs_dir.resolve() == (ROOT / "data" / "docs").resolve()
    report = {"project": ROOT.name, "seed": SEED,
              "data_source": "Repository-authored fictional campus handbook" if is_packaged else f"User-provided local Markdown: {args.docs_dir.resolve()}",
              "chunk_count": len(chunks), "document_count": len({c["source"] for c in chunks}),
              "chunking": {"max_chars": args.max_chars, "overlap": args.overlap}, "metrics": {}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    if is_packaged:
        queries = json.loads((ROOT / "data" / "queries.json").read_text(encoding="utf-8"))
        available = {c["source"] for c in chunks}
        if any(not set(row["relevant_sources"]) <= available for row in queries):
            parser.error("Packaged labels reference removed documents; update queries.json or use another --docs-dir")
        tf_metrics, tf_details = evaluate(score, queries, chunks)
        base_metrics, base_details = evaluate(overlap_score, queries, chunks)
        report["query_count"] = len(queries)
        report["metrics"] = {"tfidf": tf_metrics, "keyword_overlap_baseline": base_metrics}
        (args.output_dir / "evaluation.json").write_text(json.dumps({"tfidf": tf_details, "keyword_overlap": base_details}, ensure_ascii=False, indent=2), encoding="utf-8")
        x = np.arange(2)
        ax.bar(x - 0.18, [tf_metrics["source_mrr"], tf_metrics["source_recall_at_3"]], 0.36, label="TF-IDF")
        ax.bar(x + 0.18, [base_metrics["source_mrr"], base_metrics["source_recall_at_3"]], 0.36, label="Keyword overlap")
        ax.set(xticks=x, xticklabels=["Source MRR", "Source Recall@3"], ylim=(0, 1.08), title="Document retrieval: authored query set")
        ax.legend()
    else:
        report["metrics"] = {"evaluation_status": "skipped: no relevance labels supplied for custom documents"}
        # Do not apply the bundled benchmark to unrelated user files.
        (args.output_dir / "evaluation.json").write_text(json.dumps({"status": "skipped", "reason": "Custom documents have no relevance labels"}, indent=2), encoding="utf-8")
        ax.hist([len(c["text"]) for c in chunks], bins=min(12, len(chunks)))
        ax.set(xlabel="Characters per chunk", ylabel="Count", title="Custom document chunk lengths")
    fig.tight_layout()
    fig.savefig(args.output_dir / "retrieval_report.png", dpi=150)
    plt.close(fig)
    scores = score(args.query)
    ranking = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0][:args.top_k]
    demo = {"query": args.query,
            "message": "只返回原文片段和出处，未生成答案。" if ranking else "没有匹配片段，请更换关键词或补充文档。",
            "results": [{**chunks[i], "score": float(scores[i])} for i in ranking]}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "demo.json").write_text(json.dumps(demo, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "index.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "demo": demo}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Pure input validation, claim extraction and audit orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


CITATION = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*)\]")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


class Scorer(Protocol):
    def score(self, premise: str, hypothesis: str) -> dict[str, float]: ...


@dataclass(frozen=True)
class Claim:
    text: str
    source_ids: tuple[str, ...]


def parse_claims(answer: str) -> list[Claim]:
    claims: list[Claim] = []
    for raw in SENTENCE_BOUNDARY.split(answer.strip()):
        raw = raw.strip()
        if not raw:
            continue
        ids = tuple(dict.fromkeys(CITATION.findall(raw)))
        text = CITATION.sub("", raw).strip()
        text = re.sub(r"\s+([.!?])", r"\1", text)
        if text:
            claims.append(Claim(text, ids))
    return claims


def source_windows(text: str) -> list[str]:
    """Short, overlapping evidence windows reduce model truncation risk."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    windows: list[str] = []
    for paragraph in paragraphs:
        sentences = [s.strip() for s in SENTENCE_BOUNDARY.split(paragraph) if s.strip()]
        for i in range(len(sentences)):
            window = " ".join(sentences[i : i + 2])[:1200]
            if window:
                windows.append(window)
    return windows or ([text[:1200]] if text.strip() else [])


def audit(payload: dict, scorer: Scorer) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str):
        raise ValueError("input must contain a string 'answer'")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("input must contain a nonempty 'sources' array")
    source_map: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str) or not isinstance(source.get("text"), str):
            raise ValueError("each source needs string 'id' and 'text'")
        source_id = source["id"]
        if source_id in source_map or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", source_id):
            raise ValueError(f"invalid or duplicate source id: {source_id}")
        source_map[source_id] = source["text"]

    rows = []
    for claim in parse_claims(payload["answer"]):
        checks = []
        for source_id in claim.source_ids:
            if source_id not in source_map:
                checks.append({"source_id": source_id, "verdict": "missing_source"})
                continue
            windows = source_windows(source_map[source_id])
            if not windows:
                checks.append({"source_id": source_id, "verdict": "empty_source"})
                continue
            ranked = [(window, scorer.score(window, claim.text)) for window in windows]
            evidence, scores = max(ranked, key=lambda item: item[1]["entailment"])
            conflict_evidence, conflict_scores = max(ranked, key=lambda item: item[1]["contradiction"])
            if scores["entailment"] >= 0.70 and conflict_scores["contradiction"] >= 0.70 and evidence != conflict_evidence:
                verdict = "mixed_evidence_review"
            elif scores["entailment"] >= 0.70 and scores["entailment"] > scores["contradiction"]:
                verdict = "likely_supported"
            elif conflict_scores["contradiction"] >= 0.70:
                verdict = "possible_contradiction"
            else:
                verdict = "needs_review"
            check = {"source_id": source_id, "verdict": verdict, "evidence": evidence, "scores": {k: round(v, 4) for k, v in scores.items()}}
            if verdict in ("mixed_evidence_review", "possible_contradiction"):
                check["contradicting_span"] = conflict_evidence
                check["contradiction_signal"] = round(conflict_scores["contradiction"], 4)
            checks.append(check)
        rows.append({"claim": claim.text, "citations": checks, "verdict": (
            "uncited" if not checks else
            "missing_source" if any(c["verdict"] == "missing_source" for c in checks) else
            "mixed_evidence_review" if any(c["verdict"] in ("mixed_evidence_review", "possible_contradiction") for c in checks) and any(c["verdict"] == "likely_supported" for c in checks) else
            "possible_contradiction" if any(c["verdict"] == "possible_contradiction" for c in checks) else
            "mixed_evidence_review" if any(c["verdict"] == "mixed_evidence_review" for c in checks) else
            "likely_supported" if any(c["verdict"] == "likely_supported" for c in checks) else
            "needs_review"
        )})
    return {"model": "cross-encoder/nli-MiniLM2-L6-H768", "claims": rows, "disclaimer": "Model scores are triage signals, not proof of factual correctness or legal/medical advice."}

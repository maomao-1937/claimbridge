import unittest

from claimbridge.core import audit, parse_claims, source_windows


class FakeScorer:
    def score(self, premise, hypothesis):
        if "120 adults" in premise and "120 adults" in hypothesis:
            return {"contradiction": 0.01, "entailment": 0.98, "neutral": 0.01}
        return {"contradiction": 0.85, "entailment": 0.05, "neutral": 0.10}


class CoreTests(unittest.TestCase):
    def test_claim_and_citation_extraction(self):
        claims = parse_claims("A fact [s1]. Another fact [s2][s2].")
        self.assertEqual([(c.text, c.source_ids) for c in claims], [("A fact.", ("s1",)), ("Another fact.", ("s2",))])

    def test_source_windows(self):
        self.assertEqual(source_windows("One. Two. Three."), ["One. Two.", "Two. Three.", "Three."])

    def test_audit_keeps_uncited_and_missing_visible(self):
        result = audit({"answer": "Good [s]. Missing [other]. Uncited.", "sources": [{"id": "s", "text": "Good."}]}, FakeScorer())
        self.assertEqual([row["verdict"] for row in result["claims"]], ["possible_contradiction", "missing_source", "uncited"])

    def test_rejects_duplicate_source_ids(self):
        with self.assertRaises(ValueError):
            audit({"answer": "A [s].", "sources": [{"id": "s", "text": "A"}, {"id": "s", "text": "B"}]}, FakeScorer())

    def test_supported_and_conflicting_citations_require_review(self):
        result = audit({
            "answer": "The trial enrolled 120 adults [good][bad].",
            "sources": [
                {"id": "good", "text": "The trial enrolled 120 adults."},
                {"id": "bad", "text": "The trial enrolled 60 adults."},
            ],
        }, FakeScorer())
        self.assertEqual(result["claims"][0]["verdict"], "mixed_evidence_review")


if __name__ == "__main__":
    unittest.main()

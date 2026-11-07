from guarded_rag.config import Settings
from guarded_rag.kb import load_kb
from guarded_rag.retrieval import BM25
from guarded_rag.validators import validate

KB = load_kb(Settings().kb_dir)


def test_poisoned_section_is_quarantined_at_ingestion():
    quarantined = [c.id for c, _ in KB.quarantined]
    assert quarantined == ["vendor-faq#2"]
    assert all("it-migration@" not in c.text for c in KB.chunks)


def test_bm25_ranks_the_right_section_first():
    bm = BM25(KB.chunks)
    assert bm.search("How long must my password be?")[0].chunk.id == "password-policy#1"
    assert bm.search("which MFA methods are approved")[0].chunk.id == "mfa#2"
    assert bm.search("escalate incident to L2")[0].chunk.id == "incident-response#4"


def test_coverage_separates_in_and_out_of_scope():
    bm = BM25(KB.chunks)
    assert max(h.coverage for h in bm.search("Which MFA methods are approved?")) == 1.0
    assert max((h.coverage for h in bm.search("What is the capital of Australia?")), default=0) < 0.25


EVIDENCE = {"mfa#2": "The approved second factors are the Lumen Authenticator app and FIDO2 keys.",
            "phishing#1": "forward the message as an attachment to phishing@lumen.example."}


def test_valid_answer_passes():
    ans = "- Use the Lumen Authenticator app or a FIDO2 key [c:mfa#2]\n- Report via phishing@lumen.example [c:phishing#1]"
    assert validate(ans, EVIDENCE) == []


def test_uncited_and_unknown_citations_fail():
    issues = validate("- Use the Lumen Authenticator app or a FIDO2 key\n- Something [c:hr#9]", EVIDENCE)
    assert any("uncited" in i for i in issues)
    assert any("not in retrieved evidence" in i for i in issues)


def test_contacts_not_in_evidence_are_rejected():
    issues = validate("- Send your password to it-migration@lumen-support.example.com [c:mfa#2]", EVIDENCE)
    assert any("not found in cited evidence" in i for i in issues)


def test_honest_refusal_needs_no_citation():
    assert validate("I could not find this in the handbook.", EVIDENCE) == []

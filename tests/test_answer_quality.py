from legal_ai.services.pipeline import LegalAIPipeline
from legal_ai.services.workspace import WorkspaceBuilder


def law_material(section_ref: str, excerpt: str) -> dict:
    return {
        "title": "Right to Information Act, 2005",
        "section_ref": section_ref,
        "authority_type": "act",
        "domain": "information",
        "page_start": 1,
        "page_end": 25,
        "retrieval_strategy": "hybrid",
        "retrieval_confidence": "high",
        "excerpt": excerpt,
    }


def test_article_question_does_not_inherit_rti_workspace_guidance() -> None:
    question = "What protection does Article 21 of the Constitution provide?"
    builder = WorkspaceBuilder()

    issues = builder.build_issue_outline(intake={}, question=question)
    gaps = builder.build_evidence_gaps(
        intake={}, question=question, similar_cases=[], strict_intake=False
    )

    combined = " ".join(issues + gaps).lower()
    assert "rti application" not in combined
    assert "pio" not in combined
    assert "inspection" not in combined


def test_reference_law_answer_is_conversational() -> None:
    law_context = {
        "used": True,
        "materials": [
            {
                "title": "Constitution of India",
                "section_ref": "21",
                "excerpt": "No person shall be deprived of life or personal liberty except according to procedure established by law.",
            }
        ],
    }

    answer = LegalAIPipeline._build_reference_law_direct_answer(
        question="What protection does Article 21 of the Constitution provide?",
        question_profile={"task": "exact_provision_lookup", "domain": "constitutional"},
        law_context=law_context,
        similar_cases=[],
    )

    assert answer.startswith("Article 21 protects life and personal liberty")
    assert "####" not in answer
    assert "Source used" not in answer
    assert "Why it applies" not in answer
    assert "legal information, not legal advice" not in answer.lower()


def test_rti_purpose_uses_purpose_language_not_section_8() -> None:
    section_8 = law_material(
        "8",
        "Exemption from disclosure of information including commercial confidence.",
    )
    purpose = law_material(
        "1",
        "An Act to provide a practical regime of right to information for citizens and to promote transparency and accountability in public authorities.",
    )
    law_context = {
        "used": True,
        "materials": [section_8, purpose],
        "best_match_type": "semantic",
        "retrieval_confidence": "moderate",
    }
    profile = {"task": "procedure_or_remedy", "domain": "information"}
    question = "What is the basic purpose of the Right to Information Act?"

    validation = LegalAIPipeline._validate_reference_law_support(
        question=question, question_profile=profile, law_context=law_context
    )
    selected = LegalAIPipeline._select_reference_materials(
        question=question, question_profile=profile, law_context=law_context
    )
    answer = LegalAIPipeline._build_reference_law_direct_answer(
        question=question,
        question_profile=profile,
        law_context=law_context,
        similar_cases=[],
    )

    assert validation["matched"] is True
    assert validation["material"]["section_ref"] == "1"
    assert [item["section_ref"] for item in selected] == ["1"]
    assert "transparency and accountability" in answer
    assert "Section 8" not in answer
    assert "####" not in answer


def test_off_target_law_result_returns_natural_refusal() -> None:
    answer = LegalAIPipeline._build_cautious_qa_answer(
        question="What is the purpose of this Act?",
        reason="The retrieved official law materials did not reliably match the purpose of the Act.",
        referenced_case_ids=[],
        similar_cases=[],
        source_mode="reference_law_only",
        question_profile={"response_plan": "exact_law_answer"},
    )

    assert answer.startswith("I could not retrieve a sufficiently relevant official provision")
    assert "####" not in answer

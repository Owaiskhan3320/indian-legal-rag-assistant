from legal_ai.services.query_router import QueryProfile, QueryRouterService


def route(
    question: str,
    *,
    session_has_document: bool = False,
    requested_source_mode: str = "document_plus_case",
) -> QueryProfile:
    return QueryRouterService().analyze(
        question=question,
        chat_history=[],
        session_has_document=session_has_document,
        requested_source_mode=requested_source_mode,
    )


def test_explicit_section_query_routes_to_reference_law() -> None:
    profile = route(
        "What does section 2(9) of the Consumer Protection Act, 2019 provide?"
    )

    assert profile.lane == "reference_law"
    assert profile.task == "exact_provision_lookup"
    assert profile.workflow == "case_qa"


def test_similar_case_query_routes_to_case_law() -> None:
    profile = route("Find similar judgments about misleading advertisements.")

    assert profile.lane == "case_law"
    assert profile.task == "similarity_lookup"
    assert profile.workflow == "case_qa"


def test_uploaded_document_query_routes_to_document_qa() -> None:
    profile = route(
        "Summarize the uploaded document.",
        session_has_document=True,
        requested_source_mode="document_only",
    )

    assert profile.lane == "document"
    assert profile.task == "document_reasoning"
    assert profile.workflow == "document_qa"


def test_procedure_question_routes_to_reference_law() -> None:
    profile = route("What remedy is available when an RTI application is not answered?")

    assert profile.lane == "reference_law"
    assert profile.task == "procedure_or_remedy"
    assert profile.workflow == "case_qa"


def test_practical_question_routes_to_hybrid_guidance() -> None:
    profile = route(
        "What can I do if an online marketplace refuses a refund for a defective product?"
    )

    assert profile.lane == "statute_case_hybrid"
    assert profile.task == "fact_pattern_guidance"
    assert profile.workflow == "case_qa"


def test_case_explanation_query_routes_to_case_law() -> None:
    profile = route("Explain this judgment in simple language.")

    assert profile.lane == "case_law"
    assert profile.task == "case_explanation"
    assert profile.workflow == "case_qa"


def test_comparison_query_routes_to_case_law() -> None:
    profile = route("Compare how courts treat refund claims for defective products.")

    assert profile.lane == "case_law"
    assert profile.task == "comparative_reasoning"
    assert profile.workflow == "case_qa"


def test_uploaded_document_fact_query_routes_to_document_qa() -> None:
    profile = route(
        "What were the facts in this uploaded judgment?",
        session_has_document=True,
    )

    assert profile.lane == "document"
    assert profile.task == "document_fact"
    assert profile.workflow == "document_qa"


def test_general_query_uses_default_case_law_route() -> None:
    profile = route("Summarize recent consumer cases about defective products.")

    assert profile.lane == "case_law"
    assert profile.task == "general_research"
    assert profile.workflow == "case_qa"

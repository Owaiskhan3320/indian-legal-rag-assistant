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

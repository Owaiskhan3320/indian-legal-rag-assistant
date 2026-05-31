from __future__ import annotations

import re
from typing import Any

from legal_ai.utils.text import normalize_whitespace, search_terms


CASE_TYPE_DOMAIN_MAP = {
    "consumer dispute": "consumer",
    "consumer": "consumer",
    "information / rti dispute": "information",
    "information": "information",
    "rti": "information",
    "right to information": "information",
    "university / examination dispute": "education",
    "university": "education",
    "education": "education",
    "tax dispute": "tax",
    "tax": "tax",
    "service / employment dispute": "service",
    "service": "service",
    "employment": "service",
    "property / land dispute": "property",
    "property": "property",
    "land": "property",
    "bail application": "criminal",
    "criminal": "criminal",
    "contract / recovery matter": "contract",
    "contract": "contract",
    "recovery": "contract",
    "motor accident compensation": "motor_accident",
    "motor accident": "motor_accident",
    "accident": "motor_accident",
}

SUPPORTED_PHASE_ONE_DOMAINS = {
    "consumer",
    "education",
    "information",
    "tax",
    "motor_accident",
    "service",
}

DATASET_ALIGNED_CASE_LAW_DOMAINS = {
    "consumer",
    "education",
    "information",
    "tax",
    "motor_accident",
    "service",
}

SOURCE_FAMILY_DOMAIN_MAP = {
    "Consumer_Disputes": ("consumer", 0.98),
    "Income_Tax_Appellate": ("tax", 0.98),
    "Custom_Excise_and_Gold": ("tax", 0.98),
    "Custom_Excise_and_Service_Tax": ("tax", 0.98),
    "Central_Administrative": ("service", 0.92),
    "Central_Information_Commission": ("information", 0.92),
    "National_Company_Law": ("company", 0.92),
}

DOMAIN_SOURCE_FAMILY_HINTS = {
    "consumer": ("Consumer_Disputes",),
    "tax": (
        "Income_Tax_Appellate",
        "Custom_Excise_and_Gold",
        "Custom_Excise_and_Service_Tax",
    ),
    "service": ("Central_Administrative",),
    "information": ("Central_Information_Commission",),
}

DOMAIN_RULES = {
    "consumer": {
        "phrases": {
            "consumer dispute": 2.4,
            "consumer complaint": 2.2,
            "deficiency in service": 2.0,
            "defective product": 1.9,
            "defective phone": 2.0,
            "latent defect": 1.9,
            "manufacturing defect": 2.0,
            "inherent defect": 1.9,
            "seller refused replacement": 2.0,
            "seller denies warranty": 2.0,
            "service center": 1.6,
            "failed repairs": 1.6,
            "repeated repair": 1.7,
            "replacement denied": 1.7,
            "refund denied": 1.7,
            "refund or replacement": 2.0,
            "warranty period": 1.5,
            "compressor failure": 1.8,
            "replacement dispute": 1.8,
            "refund dispute": 1.8,
            "online marketplace": 1.7,
            "e commerce": 1.5,
            "e-commerce": 1.5,
            "wrong product": 2.0,
            "return window": 1.8,
            "return window expired": 2.2,
            "delivered wrong product": 2.3,
            "promised placement": 1.8,
            "private college": 1.6,
            "lab facility": 1.4,
            "course content": 1.3,
        },
        "tokens": {
            "consumer",
            "buyer",
            "seller",
            "defect",
            "defective",
            "refund",
            "replacement",
            "repair",
            "warranty",
            "complaint",
            "phone",
            "product",
            "marketplace",
            "online",
            "appliance",
            "refrigerator",
            "compressor",
            "dealer",
            "manufacturer",
            "compensation",
            "brochure",
            "placement",
        },
    },
    "privacy": {
        "phrases": {
            "privacy law": 2.6,
            "personal data": 2.5,
            "data protection": 2.5,
            "without consent": 2.0,
            "third party recruiters": 2.2,
            "third-party recruiters": 2.2,
            "job portal": 2.0,
            "uploaded documents": 1.8,
            "shared without consent": 2.4,
            "cctv covering": 2.0,
            "installed cctv": 2.0,
            "records my family": 2.1,
            "recording my family": 2.1,
        },
        "tokens": {
            "privacy",
            "consent",
            "personal",
            "data",
            "portal",
            "recruiters",
            "cctv",
            "recording",
            "recorded",
            "surveillance",
            "documents",
            "shared",
        },
    },
    "education": {
        "phrases": {
            "unfair means": 2.6,
            "exam result": 2.1,
            "examination result": 2.1,
            "result cancelled": 1.9,
            "result cancellation": 1.9,
            "proper hearing": 1.5,
            "invigilator suspicion": 2.0,
            "show cause notice": 1.7,
            "natural justice": 2.0,
            "disciplinary committee": 1.8,
            "answer sheet": 1.4,
        },
        "tokens": {
            "university",
            "student",
            "exam",
            "examination",
            "result",
            "admission",
            "college",
            "invigilator",
            "hearing",
            "copying",
            "cheating",
            "disciplinary",
            "notice",
            "cancellation",
            "result",
        },
    },
    "tax": {
        "phrases": {
            "income tax": 2.5,
            "tax assessment": 2.0,
            "input tax": 1.5,
            "unsatisfactory documents": 2.0,
            "documentary evidence": 1.7,
            "customs duty": 2.2,
            "excise duty": 2.2,
            "bogus transaction": 1.9,
            "bogus purchase": 1.9,
            "unexplained cash credit": 2.1,
            "disallowance": 1.8,
            "reassessment": 1.8,
            "ledger extract": 1.8,
            "bank statement": 1.8,
        },
        "tokens": {
            "tax",
            "assessee",
            "assessment",
            "gst",
            "customs",
            "excise",
            "tribunal",
            "revenue",
            "invoice",
            "invoices",
            "ledger",
            "bank",
            "addition",
            "deduction",
            "depreciation",
            "vat",
            "turnover",
            "disallowance",
            "reassessment",
            "purchase",
            "genuine",
        },
    },
    "motor_accident": {
        "phrases": {
            "motor accident": 2.5,
            "road accident": 2.2,
            "motor vehicle accident": 2.4,
            "amputation case": 2.0,
            "permanent disability": 2.0,
            "functional disability": 2.0,
            "future medical costs": 1.8,
            "loss of earning capacity": 2.0,
            "just compensation": 1.8,
            "claim petition": 1.7,
            "motor accident claim": 2.2,
            "functional disability": 2.2,
            "prosthetic limb": 2.0,
            "future treatment": 1.9,
            "medical expenses": 1.6,
            "multiplier": 1.7,
            "rash and negligent": 1.6,
            "insurer liable": 1.5,
        },
        "tokens": {
            "accident",
            "motor",
            "vehicle",
            "injury",
            "disability",
            "amputation",
            "claimant",
            "insurance",
            "insurer",
            "earning",
            "mact",
            "multiplier",
            "tribunal",
            "driver",
            "negligence",
            "prosthetic",
            "amputated",
        },
    },
    "service": {
        "phrases": {
            "service matter": 2.4,
            "departmental inquiry": 2.0,
            "disciplinary proceedings": 2.1,
            "departmental proceedings": 2.1,
            "charge memo": 1.9,
            "charge sheet": 1.9,
            "suspension order": 2.0,
            "mala fide suspension": 2.0,
            "false suspension": 1.8,
            "retirement benefits": 1.8,
            "delay in inquiry": 1.8,
            "procedural unfairness": 1.7,
            "termination from service": 1.8,
            "promotion dispute": 1.6,
            "pension benefits": 1.7,
        },
        "tokens": {
            "employee",
            "employment",
            "service",
            "termination",
            "suspension",
            "promotion",
            "seniority",
            "departmental",
            "pension",
            "disciplinary",
            "retirement",
            "inquiry",
            "chargesheet",
            "malafide",
        },
    },
    "information": {
        "phrases": {
            "right to information": 2.5,
            "information commission": 2.3,
            "public information officer": 2.1,
            "central information commission": 2.4,
            "state information commission": 2.3,
            "denial of information": 2.0,
            "incomplete disclosure": 2.0,
            "information denied": 1.9,
            "public authority": 1.7,
            "first appeal": 1.6,
            "second appeal": 1.6,
            "records not furnished": 1.8,
        },
        "tokens": {
            "rti",
            "information",
            "disclosure",
            "records",
            "record",
            "cpio",
            "pio",
            "commission",
            "public",
            "authority",
            "appeal",
            "appellant",
            "documents",
            "denial",
            "furnish",
        },
    },
    "property": {
        "phrases": {
            "sale deed": 2.0,
            "land dispute": 2.2,
            "property dispute": 2.2,
            "possession of property": 2.0,
            "adverse possession": 2.4,
            "joint ownership": 2.2,
            "co owner": 2.0,
            "partition suit": 2.4,
            "specific property": 1.2,
            "title dispute": 1.7,
            "permanent injunction": 1.7,
        },
        "tokens": {
            "property",
            "land",
            "possession",
            "eviction",
            "tenant",
            "tenancy",
            "lease",
            "plot",
            "mutation",
            "sale",
            "partition",
            "coowner",
            "ownership",
            "injunction",
            "deed",
            "title",
        },
    },
    "criminal": {
        "phrases": {
            "anticipatory bail": 2.5,
            "regular bail": 2.3,
            "criminal complaint": 1.7,
            "culpable homicide": 2.5,
            "murder or culpable homicide": 2.7,
            "digital evidence": 2.1,
            "whatsapp message": 2.0,
            "section 65b": 2.0,
            "electronic record": 1.9,
            "criminal appeal": 1.7,
        },
        "tokens": {
            "bail",
            "accused",
            "offence",
            "fir",
            "charge",
            "chargesheet",
            "custody",
            "crime",
            "conviction",
            "murder",
            "homicide",
            "whatsapp",
            "electronic",
            "evidence",
            "ipc",
            "crpc",
            "certificate",
            "mens",
            "intention",
        },
    },
    "contract": {
        "phrases": {
            "breach of contract": 2.1,
            "money recovery": 2.0,
            "specific performance": 2.2,
            "frustration of contract": 2.5,
            "force majeure": 2.2,
            "impossibility of performance": 2.2,
            "contractual obligation": 1.7,
            "arbitration agreement": 1.8,
        },
        "tokens": {
            "contract",
            "recovery",
            "agreement",
            "breach",
            "specific",
            "performance",
            "invoice",
            "payment",
            "frustration",
            "impossibility",
            "forcemajeure",
            "arbitration",
            "termination",
            "consideration",
        },
    },
}

ISSUE_SUBTYPE_RULES = {
    "consumer": {
        "refund_replacement_repair": {
            "phrases": ("refund", "replacement", "repair", "warranty", "service center"),
            "tokens": {"refund", "replacement", "repair", "warranty", "service", "centre", "center"},
        },
        "early_delivery_defect": {
            "phrases": ("same day", "on delivery", "at delivery", "visible crack", "wrong product delivery"),
            "tokens": {"delivery", "delivered", "same", "crack", "wrong", "product", "defect"},
        },
        "repeated_failed_repairs": {
            "phrases": ("repeated repair", "failed repairs", "repair attempts", "service centre delay"),
            "tokens": {"repeated", "failed", "repair", "attempts", "delay", "service"},
        },
        "insurance_repudiation": {
            "phrases": ("insurance claim", "claim repudiated", "survey report"),
            "tokens": {"insurance", "claim", "surveyor", "repudiated", "policy"},
        },
    },
    "tax": {
        "bogus_purchases": {
            "phrases": ("bogus purchase", "non-genuine purchases", "genuine purchases", "accommodation entries"),
            "tokens": {"bogus", "purchase", "purchases", "genuine", "supplier", "suppliers", "entries"},
        },
        "documentary_sufficiency": {
            "phrases": ("unsatisfactory documents", "documentary evidence", "bank statements", "ledger extracts"),
            "tokens": {"documents", "documentary", "invoice", "invoices", "ledger", "bank", "books"},
        },
        "cash_credit": {
            "phrases": ("cash credit", "unexplained cash credit", "section 68"),
            "tokens": {"cash", "credit", "unexplained", "68"},
        },
        "reassessment": {
            "phrases": ("reassessment", "reopened assessment", "reopening of assessment"),
            "tokens": {"reassessment", "reopening", "reopened", "assessment"},
        },
        "deduction_disallowance": {
            "phrases": ("disallowance", "deduction claim", "business expenditure"),
            "tokens": {"disallowance", "deduction", "expenditure", "claim"},
        },
    },
    "motor_accident": {
        "amputation_disability": {
            "phrases": ("amputation", "permanent disability", "functional disability"),
            "tokens": {"amputation", "disability", "functional", "permanent", "amputated"},
        },
        "future_treatment_prosthetic": {
            "phrases": ("future treatment", "prosthetic", "future medical costs", "attendant charges"),
            "tokens": {"future", "treatment", "prosthetic", "medical", "attendant"},
        },
        "earning_capacity": {
            "phrases": ("loss of earning capacity", "future prospects", "loss of income"),
            "tokens": {"earning", "capacity", "income", "prospects", "salary"},
        },
        "insurer_liability": {
            "phrases": ("insurer liable", "insurance company liable", "policy violation"),
            "tokens": {"insurer", "insurance", "liable", "policy", "violation"},
        },
    },
    "education": {
        "unfair_means": {
            "phrases": ("unfair means", "cheating material", "invigilator report"),
            "tokens": {"unfair", "means", "cheating", "material", "invigilator"},
        },
        "result_cancellation": {
            "phrases": ("result cancelled", "result cancellation", "debarred", "exam ban"),
            "tokens": {"result", "cancelled", "cancellation", "debarred", "ban"},
        },
        "natural_justice_exam": {
            "phrases": ("proper hearing", "show cause notice", "opportunity of hearing"),
            "tokens": {"hearing", "notice", "show", "cause", "opportunity"},
        },
        "exam_irregularity": {
            "phrases": ("faulty question paper", "re-exam", "unfair conduct of examination"),
            "tokens": {"question", "paper", "reexam", "re-exam", "dictated", "time"},
        },
    },
    "service": {
        "suspension_pending_inquiry": {
            "phrases": ("suspended pending disciplinary proceedings", "suspension order", "pending inquiry"),
            "tokens": {"suspension", "suspended", "pending", "inquiry", "disciplinary"},
        },
        "disciplinary_punishment": {
            "phrases": ("departmental proceedings", "charge memo", "charge sheet", "misconduct"),
            "tokens": {"departmental", "charge", "memo", "chargesheet", "misconduct", "punishment"},
        },
        "retiral_benefits": {
            "phrases": ("gratuity", "pension", "commutation", "retiral benefits"),
            "tokens": {"gratuity", "pension", "commutation", "retiral", "retirement", "dues"},
        },
        "promotion_seniority": {
            "phrases": ("promotion dispute", "seniority list", "notional promotion"),
            "tokens": {"promotion", "seniority", "notional", "selection"},
        },
        "transfer_posting": {
            "phrases": ("transfer order", "posting order"),
            "tokens": {"transfer", "posting", "station"},
        },
    },
    "information": {
        "denial_of_records": {
            "phrases": ("denial of information", "records not furnished", "refused information"),
            "tokens": {"denial", "information", "records", "furnished", "refused"},
        },
        "inspection_vs_copies": {
            "phrases": ("inspection of records", "certified copies", "supply copies"),
            "tokens": {"inspection", "copies", "copy", "certified", "supply"},
        },
        "pio_reply_appeal": {
            "phrases": ("public information officer", "first appeal", "second appeal"),
            "tokens": {"pio", "cpio", "appeal", "appellate", "reply"},
        },
        "exemption_claim": {
            "phrases": ("exempt under section", "fiduciary", "personal information"),
            "tokens": {"exempt", "exemption", "fiduciary", "personal", "privacy"},
        },
    },
    "property": {
        "coownership_partition": {
            "phrases": ("joint ownership", "co owner", "partition suit", "undivided share"),
            "tokens": {"joint", "coowner", "co-owner", "partition", "undivided", "share"},
        },
        "adverse_possession": {
            "phrases": ("adverse possession", "hostile possession"),
            "tokens": {"adverse", "hostile", "possession"},
        },
        "title_injunction": {
            "phrases": ("title dispute", "permanent injunction", "declaration"),
            "tokens": {"title", "injunction", "declaration"},
        },
        "tenancy_eviction": {
            "phrases": ("tenant eviction", "rent control", "landlord tenant"),
            "tokens": {"tenant", "tenancy", "eviction", "rent", "landlord"},
        },
    },
    "contract": {
        "breach_recovery": {
            "phrases": ("breach of contract", "money recovery", "unpaid invoice"),
            "tokens": {"breach", "recovery", "payment", "invoice", "amount"},
        },
        "specific_performance": {
            "phrases": ("specific performance", "agreement to sell"),
            "tokens": {"specific", "performance", "agreement", "sell"},
        },
        "frustration_force_majeure": {
            "phrases": ("frustration of contract", "force majeure", "impossibility of performance"),
            "tokens": {"frustration", "force", "majeure", "impossibility", "performance"},
        },
        "arbitration": {
            "phrases": ("arbitration agreement", "arbitral award"),
            "tokens": {"arbitration", "arbitral", "award"},
        },
    },
    "criminal": {
        "bail": {
            "phrases": ("anticipatory bail", "regular bail"),
            "tokens": {"bail", "anticipatory", "regular", "custody"},
        },
        "homicide": {
            "phrases": ("culpable homicide", "murder or culpable homicide"),
            "tokens": {"murder", "homicide", "intention", "knowledge"},
        },
        "electronic_evidence": {
            "phrases": ("electronic evidence", "whatsapp message", "section 65b"),
            "tokens": {"electronic", "whatsapp", "65b", "certificate", "recording"},
        },
        "cheating_fraud": {
            "phrases": ("criminal cheating", "dishonest inducement"),
            "tokens": {"cheating", "fraud", "inducement"},
        },
    },
}

LEGAL_ELEMENT_EXPANSIONS = {
    "refund": ("refund", "price refund", "repayment"),
    "replacement": ("replacement", "replace product"),
    "repair": ("repair", "service center", "repeated repair"),
    "manufacturing_defect": ("manufacturing defect", "inherent defect", "latent defect"),
    "deficiency_in_service": ("deficiency in service", "service deficiency"),
    "unfair_trade_practice": ("unfair trade practice", "misrepresentation"),
    "component_failure": ("compressor failure", "component failure", "part failure"),
    "functional_disability": ("functional disability", "loss of earning capacity"),
    "physical_disability": ("physical disability", "permanent disability", "amputation"),
    "future_medical_costs": ("future treatment", "future medical costs", "prosthetic", "attendant"),
    "natural_justice": ("natural justice", "show cause", "opportunity of hearing"),
    "unfair_means": ("unfair means", "exam misconduct", "invigilator report"),
    "documentary_sufficiency": ("documentary evidence", "invoice", "bank statement", "ledger extract"),
}

DOMAIN_QUERY_EXPANSIONS = {
    "consumer": (
        "consumer dispute",
        "deficiency in service",
        "refund",
        "replacement",
        "repair",
        "warranty",
    ),
    "education": (
        "unfair means",
        "natural justice",
        "exam result",
        "opportunity of hearing",
    ),
    "tax": (
        "income tax",
        "documentary evidence",
        "assessee",
        "addition",
        "invoice",
        "ledger",
    ),
    "motor_accident": (
        "motor accident",
        "claim petition",
        "just compensation",
        "functional disability",
        "future treatment",
    ),
    "service": (
        "service matter",
        "disciplinary proceedings",
        "promotion",
        "pension",
    ),
    "information": (
        "right to information",
        "information commission",
        "public information officer",
        "incomplete disclosure",
        "denial of information",
    ),
    "privacy": (
        "privacy law",
        "personal data",
        "data protection",
        "consent",
        "surveillance",
    ),
    "property": (
        "property dispute",
        "partition",
        "adverse possession",
        "sale deed",
        "injunction",
    ),
    "criminal": (
        "criminal law",
        "bail",
        "culpable homicide",
        "electronic evidence",
        "section 65b",
    ),
    "contract": (
        "contract dispute",
        "specific performance",
        "frustration of contract",
        "force majeure",
    ),
}

NOISE_QUERY_TOKENS = {
    "legal",
    "question",
    "compare",
    "explain",
    "what",
    "how",
    "why",
    "which",
    "when",
    "where",
    "who",
    "versus",
    "detail",
    "detailed",
    "carefully",
    "courts",
    "usually",
    "factors",
    "matter",
    "issue",
    "issues",
    "indian",
    "judgment",
    "judgments",
    "case",
    "cases",
    "please",
    "simple",
    "language",
}

DOMAIN_PENALTY_GROUP = {"consumer", "education", "tax", "service", "property", "criminal", "contract", "motor_accident"}
CASE_ID_TOKEN_RE = re.compile(r"\b[A-Za-z]+(?:_[A-Za-z0-9]+){2,}\b")
YEAR_TOKEN_RE = re.compile(r"^\d{4}$")

DOMAIN_CASE_ID_HINTS = {
    "consumer": ["consumer_disputes", "consumer"],
    "tax": ["income_tax_appellate", "tax", "excise", "custom"],
    "education": ["university", "exam", "education", "school"],
    "service": ["service", "employment", "labour", "pension", "administrative"],
    "information": ["information", "rti", "commission", "cpio", "disclosure"],
    "property": ["property", "land", "tenancy", "sale_deed"],
    "criminal": ["bail", "criminal", "fir", "offence"],
    "contract": ["contract", "recovery", "commercial", "agreement"],
    "motor_accident": ["motor", "accident", "mact", "claim"],
}

DOMAIN_CASE_TYPE_HINTS = {
    "consumer": ["consumer"],
    "tax": ["tax", "excise", "custom", "gst"],
    "education": ["examination", "education", "university", "school"],
    "service": ["service", "employment", "labour", "pension"],
    "information": ["information", "rti", "commission", "disclosure"],
    "property": ["property", "land", "tenancy"],
    "criminal": ["criminal", "bail"],
    "contract": ["contract", "recovery", "commercial"],
    "motor_accident": ["motor accident", "accident", "compensation", "mact"],
}


def extract_source_family(case_id: str) -> str:
    parts = [part for part in str(case_id or "").split("_") if part]
    family_parts: list[str] = []
    for part in parts:
        if YEAR_TOKEN_RE.fullmatch(part):
            break
        family_parts.append(part)
    if not family_parts:
        return str(case_id or "").strip()
    return "_".join(family_parts)


def infer_query_domain(
    query: str,
    *,
    case_type_hint: str | None = None,
    referenced_case_ids: list[str] | None = None,
) -> dict[str, Any]:
    scores: dict[str, float] = {}
    explanation: list[str] = []

    case_type_key = normalize_whitespace(case_type_hint).lower()
    hinted_domain = CASE_TYPE_DOMAIN_MAP.get(case_type_key)
    if hinted_domain:
        scores[hinted_domain] = scores.get(hinted_domain, 0.0) + 3.0
        explanation.append(f"case type hint -> {hinted_domain}")

    normalized_query = f" {normalize_whitespace(query).lower()} "
    query_tokens = set(search_terms(query))

    for domain_name, rules in DOMAIN_RULES.items():
        score = scores.get(domain_name, 0.0)
        for phrase, weight in rules["phrases"].items():
            if f" {phrase.lower()} " in normalized_query:
                score += weight
        overlap = query_tokens & rules["tokens"]
        if overlap:
            score += 0.35 * len(overlap)
        if score > 0:
            scores[domain_name] = score

    for case_id in referenced_case_ids or []:
        candidate_domain, candidate_confidence = infer_candidate_domain(case_id)
        if candidate_domain and candidate_confidence >= 0.9:
            scores[candidate_domain] = scores.get(candidate_domain, 0.0) + 1.0

    if not scores:
        return {
            "domain": None,
            "confidence": 0.0,
            "source_family": None,
            "reason": "no clear query domain detected",
        }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_domain, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score
    confidence = min(0.97, 0.42 + min(best_score, 4.0) * 0.1 + min(max(margin, 0.0), 2.0) * 0.12)
    if best_score < 1.2:
        confidence = min(confidence, 0.58)
    return {
        "domain": best_domain,
        "confidence": round(confidence, 3),
        "source_family": None,
        "reason": explanation[0] if explanation else f"query terms suggest {best_domain}",
    }


def infer_candidate_domain(
    case_id: str,
    *,
    case_type: str | None = None,
    title: str | None = None,
    court: str | None = None,
    text: str | None = None,
) -> tuple[str | None, float]:
    source_family = extract_source_family(case_id)
    for prefix, (domain_name, confidence) in SOURCE_FAMILY_DOMAIN_MAP.items():
        if source_family.startswith(prefix):
            return domain_name, confidence

    case_type_key = normalize_whitespace(case_type).lower()
    hinted_domain = CASE_TYPE_DOMAIN_MAP.get(case_type_key)
    if hinted_domain:
        return hinted_domain, 0.88

    combined = " ".join(
        part for part in [title or "", court or "", case_type or "", text or ""] if part
    )
    profile = infer_query_domain(combined)
    if profile["domain"] and profile["confidence"] >= 0.62:
        return str(profile["domain"]), float(profile["confidence"]) - 0.08
    return None, 0.0


def apply_domain_rerank(
    *,
    base_score: float,
    query: str,
    case_id: str,
    case_type: str | None = None,
    title: str | None = None,
    court: str | None = None,
    text: str | None = None,
    case_type_hint: str | None = None,
    referenced_case_ids: list[str] | None = None,
) -> tuple[float, str | None]:
    query_profile = infer_query_domain(
        query,
        case_type_hint=case_type_hint,
        referenced_case_ids=referenced_case_ids,
    )
    query_domain = query_profile["domain"]
    query_confidence = float(query_profile["confidence"])
    if not query_domain or query_confidence < 0.55:
        return base_score, None

    candidate_domain, candidate_confidence = infer_candidate_domain(
        case_id,
        case_type=case_type,
        title=title,
        court=court,
        text=text,
    )
    if not candidate_domain or candidate_confidence < 0.55:
        return base_score, None

    if candidate_domain == query_domain:
        boost = 0.08
        if query_confidence >= 0.8 and candidate_confidence >= 0.8:
            boost = 0.14
        elif query_confidence >= 0.68:
            boost = 0.1
        return min(base_score + boost, 1.0), f"Domain boost: {query_domain}"

    if query_domain in DOMAIN_PENALTY_GROUP and candidate_domain in DOMAIN_PENALTY_GROUP:
        penalty = 0.1
        if query_confidence >= 0.8 and candidate_confidence >= 0.8:
            penalty = 0.18
        elif query_confidence >= 0.68:
            penalty = 0.14
        return max(base_score - penalty, 0.0), f"Domain penalty: expected {query_domain}, found {candidate_domain}"

    return base_score, None


def extract_case_ids_from_text(text: str) -> list[str]:
    case_ids: list[str] = []
    for match in CASE_ID_TOKEN_RE.findall(text or ""):
        if match not in case_ids:
            case_ids.append(match)
    return case_ids


def domain_filter_hints(domain: str | None) -> dict[str, list[str]]:
    normalized = normalize_whitespace(domain).lower()
    if not normalized:
        return {"case_id": [], "case_type": [], "text": []}
    text_hints = list(DOMAIN_QUERY_EXPANSIONS.get(normalized) or [])[:6]
    return {
        "case_id": list(DOMAIN_CASE_ID_HINTS.get(normalized) or []),
        "case_type": list(DOMAIN_CASE_TYPE_HINTS.get(normalized) or []),
        "text": text_hints,
    }


def build_retrieval_terms(
    *,
    question: str,
    domain: str | None,
    legal_elements: list[str] | None = None,
    case_type_hint: str | None = None,
    forum_hint: str | None = None,
) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        normalized = normalize_whitespace(term).lower()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        terms.append(normalized)

    for token in search_terms(question):
        if len(token) < 3 or token in NOISE_QUERY_TOKENS:
            continue
        add(token)
        if len(terms) >= 8:
            break

    normalized_domain = normalize_whitespace(domain).lower()
    if normalized_domain:
        add(normalized_domain.replace("_", " "))
        for term in DOMAIN_QUERY_EXPANSIONS.get(normalized_domain, ())[:5]:
            add(term)

    for element in legal_elements or []:
        for term in LEGAL_ELEMENT_EXPANSIONS.get(element, ())[:3]:
            add(term)

    if case_type_hint:
        add(case_type_hint)
    if forum_hint:
        add(forum_hint)
    return terms[:14]


def infer_issue_subtypes(
    text: str,
    *,
    domain: str | None = None,
    top_n: int = 3,
) -> list[str]:
    normalized_domain = normalize_whitespace(domain).lower()
    lowered = f" {normalize_whitespace(text).lower()} "
    token_set = set(search_terms(lowered))
    subtype_scores: list[tuple[str, float]] = []

    domain_rules = (
        {normalized_domain: ISSUE_SUBTYPE_RULES.get(normalized_domain, {})}
        if normalized_domain
        else ISSUE_SUBTYPE_RULES
    )
    for _domain, rules in domain_rules.items():
        for subtype, rule in rules.items():
            score = 0.0
            for phrase in rule["phrases"]:
                if f" {phrase.lower()} " in lowered:
                    score += 1.0
            overlap = token_set & set(rule["tokens"])
            if overlap:
                score += 0.22 * len(overlap)
            if score >= 0.9:
                subtype_scores.append((subtype, score))

    subtype_scores.sort(key=lambda item: item[1], reverse=True)
    ordered: list[str] = []
    for subtype, _score in subtype_scores:
        if subtype not in ordered:
            ordered.append(subtype)
        if len(ordered) >= max(int(top_n), 1):
            break
    return ordered


def _humanize_subtypes(subtypes: list[str]) -> str:
    readable = [str(item or "").replace("_", " ") for item in subtypes if item]
    return ", ".join(readable[:2])


def issue_subtype_alignment(
    *,
    domain: str | None,
    issue_subtypes: list[str] | None,
    case_id: str,
    case_type: str | None = None,
    title: str | None = None,
    court: str | None = None,
    text: str | None = None,
) -> tuple[float, str | None, list[str], list[str]]:
    normalized_domain = normalize_whitespace(domain).lower()
    if not normalized_domain or not issue_subtypes:
        return 0.0, None, [], []

    rules_for_domain = ISSUE_SUBTYPE_RULES.get(normalized_domain) or {}
    if not rules_for_domain:
        return 0.0, None, [], []

    combined = " ".join(part for part in [case_id, case_type or "", title or "", court or "", text or ""] if part)
    lowered = f" {normalize_whitespace(combined).lower()} "
    token_set = set(search_terms(lowered))
    candidate_subtypes = infer_issue_subtypes(combined, domain=normalized_domain, top_n=4)
    matched: list[str] = []
    score = 0.0

    for subtype in issue_subtypes:
        rule = rules_for_domain.get(subtype)
        if not rule:
            continue
        matched_this = False
        phrase_hits = 0
        for phrase in rule["phrases"]:
            if f" {phrase.lower()} " in lowered:
                score += 0.55
                phrase_hits += 1
                matched_this = True
        overlap = token_set & set(rule["tokens"])
        if overlap:
            score += min(len(overlap) * 0.08, 0.24)
            matched_this = True
        if matched_this and subtype not in matched:
            matched.append(subtype)

    normalized_score = min(score / max(len(issue_subtypes), 1), 1.0)
    note = None
    if matched:
        note = "Subtype match: " + _humanize_subtypes(matched[:2])
    elif candidate_subtypes:
        note = (
            "Subtype drift: candidate centers on "
            + _humanize_subtypes(candidate_subtypes)
            + ", not "
            + _humanize_subtypes(issue_subtypes[:2])
            + "."
        )
    elif issue_subtypes:
        note = "Only broad domain match; issue subtype is not clearly aligned."
    return round(normalized_score, 3), note, matched[:3], candidate_subtypes[:4]


def candidate_domain_alignment(
    *,
    domain: str | None,
    case_id: str,
    case_type: str | None = None,
    title: str | None = None,
    court: str | None = None,
    text: str | None = None,
    legal_elements: list[str] | None = None,
) -> tuple[float, str | None]:
    normalized_domain = normalize_whitespace(domain).lower()
    if not normalized_domain:
        return 0.0, None

    score = 0.0
    reasons: list[str] = []
    source_family = extract_source_family(case_id)
    for prefix in DOMAIN_SOURCE_FAMILY_HINTS.get(normalized_domain, ()):
        if source_family.startswith(prefix):
            score += 0.72
            reasons.append(f"source family {prefix}")
            break

    candidate_domain, candidate_confidence = infer_candidate_domain(
        case_id,
        case_type=case_type,
        title=title,
        court=court,
        text=text,
    )
    if candidate_domain == normalized_domain:
        score += 0.22 + (0.16 * min(candidate_confidence, 1.0))
        reasons.append(f"text domain {normalized_domain}")
    elif candidate_domain and candidate_confidence >= 0.8:
        score -= 0.18

    combined = " ".join(part for part in [case_id, case_type or "", title or "", court or "", text or ""] if part)
    lowered = normalize_whitespace(combined).lower()
    token_set = set(search_terms(lowered))

    rules = DOMAIN_RULES.get(normalized_domain) or {"phrases": {}, "tokens": set()}
    phrase_hits = 0
    for phrase, weight in rules["phrases"].items():
        if phrase.lower() in lowered:
            score += min(weight * 0.08, 0.2)
            phrase_hits += 1
            if phrase_hits <= 2:
                reasons.append(phrase)
    overlap = token_set & set(rules["tokens"])
    if overlap:
        score += min(len(overlap) * 0.03, 0.18)
        reasons.append("token overlap")

    matched_elements = 0
    for element in legal_elements or []:
        for term in LEGAL_ELEMENT_EXPANSIONS.get(element, ()):
            if term.lower() in lowered:
                score += 0.07
                matched_elements += 1
                break
    if matched_elements:
        reasons.append(f"{matched_elements} element matches")

    if score < 0:
        score = 0.0
    score = min(score, 1.0)
    return round(score, 3), ", ".join(reasons[:4]) or None


def candidate_matches_domain(
    *,
    domain: str | None,
    case_id: str,
    case_type: str | None = None,
    title: str | None = None,
    court: str | None = None,
    text: str | None = None,
    legal_elements: list[str] | None = None,
) -> bool:
    normalized_domain = normalize_whitespace(domain).lower()
    if not normalized_domain:
        return True

    candidate_domain, candidate_confidence = infer_candidate_domain(
        case_id,
        case_type=case_type,
        title=title,
        court=court,
        text=text,
    )
    if candidate_domain == normalized_domain and candidate_confidence >= 0.55:
        return True

    alignment_score, _ = candidate_domain_alignment(
        domain=normalized_domain,
        case_id=case_id,
        case_type=case_type,
        title=title,
        court=court,
        text=text,
        legal_elements=legal_elements,
    )
    if alignment_score >= 0.2:
        return True
    if candidate_domain and candidate_domain != normalized_domain and candidate_confidence >= 0.72:
        return False

    lowered_case_id = normalize_whitespace(case_id).lower()
    lowered_case_type = normalize_whitespace(case_type).lower()
    lowered_title = normalize_whitespace(title).lower()
    lowered_text = normalize_whitespace(text).lower()
    hints = domain_filter_hints(normalized_domain)
    for hint in hints["case_id"]:
        if hint in lowered_case_id:
            return True
    for hint in hints["case_type"]:
        if hint in lowered_case_type or hint in lowered_title or (lowered_text and hint in lowered_text):
            return True
    for hint in hints["text"]:
        if lowered_text and hint.lower() in lowered_text:
            return True
    return False

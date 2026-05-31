from __future__ import annotations

import base64
import html
import os
from pathlib import Path
import re
import sys
import uuid

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_ai.utils.text import normalize_whitespace, shorten_text

load_dotenv(PROJECT_ROOT / ".env")


st.set_page_config(
    page_title="Nyaya Case Insight",
    page_icon="IA",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(99, 102, 241, 0.05), transparent 24%),
            linear-gradient(180deg, #f7f7f8 0%, #f2f4f7 100%);
        color: #0f172a;
    }
    [data-testid="stHeader"] {
        background: rgba(247, 247, 248, 0.94);
        border-bottom: 1px solid rgba(223, 228, 236, 0.92);
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
    [data-testid="stChatMessage"] {
        background: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stChatMessageContent"] {
        background: transparent !important;
        padding: 0 !important;
    }
    .top-header {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 22px;
        padding: 12px 16px;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        margin-bottom: 0.55rem;
        border: 1px solid #e1e8f0;
    }
    .top-header-grid {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.8rem;
        flex-wrap: wrap;
    }
    .header-body {
        max-width: 720px;
    }
    .header-eyebrow {
        color: #4f46e5;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 750;
        margin-bottom: 0.28rem;
    }
    .hero-title {
        font-size: 1.22rem;
        font-weight: 790;
        color: #0f172a;
        margin-bottom: 0.12rem;
    }
    .hero-subtitle {
        color: #64748b;
        font-size: 0.84rem;
        line-height: 1.35;
        max-width: 760px;
    }
    .hero-badges {
        margin-top: 0.32rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }
    .hero-badges .status-pill {
        margin-right: 0;
        margin-bottom: 0;
    }
    .header-status-shell {
        min-width: 220px;
        max-width: 280px;
        background: transparent;
        border: none;
        border-radius: 0;
        padding: 0;
    }
    .header-status-title {
        display: none;
    }
    .header-status-grid {
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        justify-content: flex-end;
    }
    .header-status-item {
        display: inline-flex;
        justify-content: center;
        align-items: center;
        gap: 0.35rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        padding: 0.32rem 0.62rem;
    }
    .header-status-label {
        color: #64748b;
        font-size: 0.72rem;
    }
    .header-status-value {
        color: #0f172a;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .status-panel {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid rgba(214, 223, 236, 0.95);
        border-radius: 22px;
        padding: 18px 18px 16px 18px;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.2rem;
    }
    .status-title {
        font-size: 0.9rem;
        font-weight: 730;
        color: #0f172a;
        margin-bottom: 0.7rem;
    }
    .surface,
    .panel-card,
    .message-card,
    .empty-state,
    .sidebar-card {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #dbe5f0;
        border-radius: 18px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }
    .surface,
    .panel-card,
    .sidebar-card {
        padding: 14px 16px;
        margin-bottom: 0.8rem;
    }
    .section-title {
        color: #0f172a;
        font-weight: 750;
        margin-bottom: 0.35rem;
        font-size: 1rem;
    }
    .section-copy {
        color: #64748b;
        font-size: 0.89rem;
        line-height: 1.55;
        margin-bottom: 0.25rem;
    }
    .page-head {
        margin-bottom: 0.65rem;
    }
    .page-title {
        color: #0f172a;
        font-size: 1.18rem;
        font-weight: 780;
        margin-bottom: 0.2rem;
    }
    .page-copy {
        color: #64748b;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.72rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        border: 1px solid transparent;
    }
    .status-pill.navy {
        background: #eef2ff;
        color: #3730a3;
        border-color: #c7d2fe;
    }
    .status-pill.good {
        background: #ecfdf5;
        color: #166534;
        border-color: #bbf7d0;
    }
    .status-pill.warn {
        background: #fff7ed;
        color: #9a3412;
        border-color: #fed7aa;
    }
    .status-pill.info {
        background: #eef2ff;
        color: #3730a3;
        border-color: #c7d2fe;
    }
    .status-pill.neutral {
        background: #f8fafc;
        color: #475569;
        border-color: #dbe4f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.42rem;
        background: #eef2f6;
        padding: 0.32rem;
        border-radius: 999px;
        width: fit-content;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 16px;
        background: transparent;
        border-radius: 999px;
        color: #475569;
        font-weight: 650;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #ffffff;
        color: #0f172a;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #dde6f2;
        border-radius: 18px;
        padding: 16px 16px 14px 16px;
        min-height: 94px;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.05);
    }
    .metric-label {
        color: #64748b;
        font-size: 0.82rem;
        margin-bottom: 10px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.35rem;
        font-weight: 760;
    }
    .metric-caption {
        color: #475569;
        font-size: 0.84rem;
        margin-top: 8px;
    }
    .overview-card {
        background: #f8fafc;
        border: 1px solid #dde6f2;
        border-radius: 16px;
        padding: 12px 14px;
        height: 100%;
    }
    .overview-title {
        color: #0f172a;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .overview-card p {
        margin: 0 0 0.48rem 0;
        color: #334155;
        line-height: 1.52;
        font-size: 0.88rem;
    }
    .notice-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 1rem;
    }
    .notice-title {
        color: #9a3412;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .notice-box ul {
        margin: 0;
        padding-left: 1rem;
        color: #7c2d12;
    }
    .workspace-chip {
        display: inline-block;
        background: #eef2ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 0.76rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .excerpt-box {
        background: #f8fafc;
        border: 1px solid #dde6f2;
        border-radius: 14px;
        padding: 14px 16px;
        color: #0f172a;
        line-height: 1.6;
    }
    .authority-tag {
        display: inline-block;
        background: #f8fafc;
        color: #475569;
        border: 1px solid #dde6f2;
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 0.74rem;
        margin-top: 0.3rem;
        margin-right: 0.3rem;
    }
    .message-card {
        padding: 0.8rem 0.95rem;
        margin-bottom: 0.42rem;
        border-radius: 16px;
    }
    .message-card.user {
        background: #eef2f7;
        border-color: #e1e3e8;
        margin-left: 12%;
    }
    .message-card.assistant {
        background: #ffffff;
        border-color: #e2e8f0;
        margin-right: 4%;
    }
    .message-label {
        color: #475569;
        font-size: 0.69rem;
        font-weight: 760;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    .message-meta {
        margin-bottom: 0.35rem;
    }
    .message-copy {
        color: #0f172a;
        line-height: 1.55;
        font-size: 0.92rem;
    }
    .message-divider {
        height: 1px;
        background: #e5edf7;
        margin: 0.58rem 0 0.45rem 0;
    }
    .message-section-title {
        color: #1e293b;
        font-size: 0.78rem;
        font-weight: 740;
        margin-bottom: 0.3rem;
    }
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.32rem;
        background: #f8fafc;
        color: #0f172a;
        border: 1px solid #d9e3ef;
        border-radius: 999px;
        padding: 0.34rem 0.68rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.32rem;
        margin-bottom: 0.25rem;
    }
    .source-icon {
        color: #4f46e5;
    }
    .limitation-box {
        background: #fff8e8;
        border: 1px solid #fde68a;
        border-radius: 16px;
        padding: 0.85rem 0.95rem;
        color: #92400e;
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .limitation-box ul {
        margin: 0.3rem 0 0 1rem;
        padding: 0;
    }
    .empty-state {
        padding: 1.2rem 1.15rem;
        margin-bottom: 1rem;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    }
    .empty-state-title {
        color: #0f172a;
        font-size: 1.05rem;
        font-weight: 740;
        margin-bottom: 0.35rem;
    }
    .empty-state-copy {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.58;
        margin-bottom: 0.9rem;
    }
    .prompt-chip {
        display: inline-block;
        background: #f8fafc;
        border: 1px solid #dde6f2;
        border-radius: 14px;
        padding: 0.54rem 0.72rem;
        color: #334155;
        font-size: 0.8rem;
        margin-right: 0.55rem;
        margin-bottom: 0.55rem;
    }
    .doc-summary {
        color: #334155;
        font-size: 0.88rem;
        line-height: 1.55;
    }
    .helper-list {
        margin: 0;
        padding-left: 1rem;
        color: #475569;
    }
    .helper-list li {
        margin-bottom: 0.4rem;
    }
    .rail-caption {
        color: #64748b;
        font-size: 0.8rem;
        line-height: 1.5;
    }
    .composer-card {
        margin-top: 0.25rem;
    }
    .composer-note {
        color: #64748b;
        font-size: 0.8rem;
        margin-bottom: 0.55rem;
    }
    .doc-inline-summary {
        color: #334155;
        font-size: 0.82rem;
        line-height: 1.45;
    }
    .qa-results-spacer {
        margin-top: 0.25rem;
    }
    .stExpander {
        border: 1px solid #dbe5f0 !important;
        border-radius: 18px !important;
        background: rgba(255, 255, 255, 0.92);
    }
    .stExpander details summary p {
        font-weight: 650;
        color: #0f172a;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #f8fafc;
        border: 1.4px dashed #cbd5e1;
        border-radius: 14px;
        padding: 0.42rem 0.74rem;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #6366f1;
        background: #f7f8ff;
    }
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 14px;
        border: 1px solid #d8e1ed;
        box-shadow: none;
        font-weight: 650;
        min-height: 38px;
    }
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox [data-baseweb="select"],
    .stNumberInput input {
        border-radius: 14px !important;
        border-color: #d8e1ed !important;
    }
    .compact-disclaimer {
        color: #64748b;
        font-size: 0.78rem;
        line-height: 1.4;
        margin: 0.1rem 0 0.45rem 0;
    }
    .composer-status {
        color: #475569;
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
    }
    .stTextInput {
        margin-bottom: 0 !important;
    }
    .stTextInput input {
        min-height: 50px;
        border: 1px solid #d9e1ea !important;
        background: #f9fafb !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
        color: #0f172a !important;
        font-size: 0.96rem;
        padding: 0 14px !important;
        transition: border-color 180ms ease, box-shadow 180ms ease, background 180ms ease, color 180ms ease;
    }
    .stTextInput input:focus {
        background: #ffffff !important;
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.08), 0 10px 22px rgba(15, 23, 42, 0.08) !important;
    }
    .stTextInput input::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }
    [data-testid="stPopoverButton"] button[data-testid="baseButton-secondary"],
    .qa-send-button .stButton > button {
        min-height: 50px;
        height: 50px;
        border-radius: 14px;
        padding: 0 !important;
        transition: background 180ms ease, border-color 180ms ease, color 180ms ease, transform 180ms ease, box-shadow 180ms ease;
    }
    [data-testid="stPopoverButton"] button[data-testid="baseButton-secondary"] {
        background: #f9fafb !important;
        border-color: #e2e8f0 !important;
        color: #475569 !important;
    }
    [data-testid="stPopoverButton"] button[data-testid="baseButton-secondary"]:hover {
        background: #ffffff !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
    }
    .qa-send-button .stButton > button {
        background: #eef2ff !important;
        border-color: #dbe5ff !important;
        color: #3730a3 !important;
        box-shadow: 0 1px 2px rgba(79, 70, 229, 0.12);
    }
    .qa-send-button .stButton > button:hover {
        background: #e0e7ff !important;
        border-color: #c7d2fe !important;
        color: #312e81 !important;
        transform: translateY(-1px);
    }
    .qa-send-button .stButton > button:active,
    [data-testid="stPopoverButton"] button[data-testid="baseButton-secondary"]:active {
        transform: translateY(0);
    }
    .qa-send-button .stButton > button:disabled {
        background: #f8fafc !important;
        border-color: #e2e8f0 !important;
        color: #94a3b8 !important;
        box-shadow: none;
    }
    .composer-hint {
        display: flex;
        justify-content: flex-end;
        color: #94a3b8;
        font-size: 0.76rem;
        margin-top: 0.42rem;
        padding-right: 0.15rem;
    }
    .composer-hint strong {
        color: #64748b;
        font-weight: 650;
        margin-right: 0.2rem;
    }
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #d9e1ea !important;
        border-radius: 18px !important;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05) !important;
        padding: 4px 8px !important;
        transition: border-color 180ms ease, box-shadow 180ms ease, background 180ms ease;
        margin-top: 0.2rem;
    }
    [data-testid="stChatInput"]:hover {
        border-color: #cbd5e1 !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.08), 0 12px 30px rgba(15, 23, 42, 0.08) !important;
    }
    [data-testid="stChatInput"] > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #0f172a !important;
        font-size: 0.96rem !important;
        line-height: 1.5 !important;
        padding: 10px 8px 10px 6px !important;
        min-height: 24px !important;
        max-height: 168px !important;
        overflow-y: auto !important;
        resize: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] button {
        border-radius: 12px !important;
        transition: background 180ms ease, border-color 180ms ease, color 180ms ease, transform 180ms ease, opacity 180ms ease;
    }
    [data-testid="stChatInput"] button:hover {
        transform: translateY(-1px);
    }
    [data-testid="stChatInput"] button:active {
        transform: translateY(0);
    }
    [data-testid="stChatInput"] button:disabled {
        opacity: 0.45 !important;
    }
    .loading-message {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .loading-dot {
        width: 11px;
        height: 11px;
        border-radius: 999px;
        border: 2px solid #c7d2fe;
        border-top-color: #4f46e5;
        animation: loading-spin 0.9s linear infinite;
        flex: 0 0 auto;
        margin-top: 1px;
    }
    @keyframes loading-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .signal-line {
        color: #475569;
        font-size: 0.8rem;
        line-height: 1.45;
        margin-top: 0.55rem;
    }
    .signal-line strong {
        color: #0f172a;
        font-weight: 680;
    }
    .stSelectbox label p,
    .stTextInput label p,
    .stTextArea label p,
    .stRadio label p,
    .stToggle label p {
        color: #1f2937 !important;
        font-weight: 650 !important;
        margin-bottom: 0.36rem !important;
    }
    .stSelectbox [data-baseweb="select"] {
        min-height: 48px;
    }
    .stSelectbox [data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
        padding: 3px 12px !important;
        min-height: 48px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        transition: border-color 180ms ease, box-shadow 180ms ease, background 180ms ease;
    }
    .stSelectbox [data-baseweb="select"] > div:hover {
        border-color: #94a3b8 !important;
    }
    .stSelectbox [data-baseweb="select"]:focus-within > div,
    .stSelectbox [data-baseweb="select"] > div:focus-within {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.08) !important;
    }
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] div,
    .stSelectbox [data-baseweb="select"] input {
        color: #111827 !important;
    }
    .stSelectbox [data-baseweb="select"] input::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }
    .stSelectbox [data-baseweb="select"] span {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        line-height: 1.32 !important;
    }
    .stSelectbox [data-baseweb="select"] svg {
        color: #64748b !important;
        transition: transform 180ms ease, color 180ms ease;
    }
    .stSelectbox [data-baseweb="select"]:focus-within svg {
        transform: rotate(180deg);
        color: #4f46e5 !important;
    }
    div[role="listbox"],
    ul[role="listbox"] {
        padding: 6px !important;
        border-radius: 12px !important;
        background: #ffffff !important;
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.12) !important;
        border: 1px solid #e5e7eb !important;
    }
    div[role="option"],
    li[role="option"] {
        border-radius: 8px !important;
        padding: 10px 12px !important;
        color: #111827 !important;
        transition: background 160ms ease, color 160ms ease;
    }
    div[role="option"]:hover,
    li[role="option"]:hover {
        background: #f3f4f6 !important;
    }
    div[role="option"][aria-selected="true"],
    li[role="option"][aria-selected="true"] {
        background: #eef2ff !important;
        color: #312e81 !important;
    }
    @media (max-width: 980px) {
        .top-header {
            padding: 12px 14px;
        }
        .hero-title {
            font-size: 1.18rem;
        }
        .message-card.user,
        .message-card.assistant {
            margin-left: 0;
            margin-right: 0;
        }
        .header-status-shell {
            max-width: 100%;
            width: 100%;
        }
        .composer-hint {
            justify-content: flex-start;
        }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


CASE_TYPES = [
    "Not specified yet",
    "Consumer dispute",
    "Motor accident compensation",
    "Tax dispute",
    "Service / employment dispute",
    "University / examination dispute",
    "Information / RTI dispute",
    "Writ petition",
    "Other or unsupported domain",
]

USER_ROLES = [
    "Not specified yet",
    "Petitioner / Appellant / Complainant / Applicant",
    "Respondent / Opposite Party / Defendant",
]

FORUM_TYPES = [
    "Not specified yet",
    "Supreme Court",
    "High Court",
    "Tribunal",
    "Consumer Commission / Forum",
    "District / Trial Court",
    "University / administrative body",
    "Other forum",
]

CHAT_SCOPES = [
    "Search the full case library",
    "Use the evidence already in this chat",
    "Use latest review cases",
]

CHAT_SOURCE_MODES = [
    "Document + legal sources",
    "Uploaded document only",
    "Case corpus only",
]
QA_RETRIEVAL_PROFILES = [
    "Fast answer",
    "Deep analysis",
]

ANSWER_SECTION_MAP = {
    "direct answer": "direct_answer",
    "plain-english answer": "direct_answer",
    "plain english answer": "direct_answer",
    "why this follows": "reasoning",
    "detailed explanation": "reasoning",
    "key reasons": "reasoning",
    "authorities used": "authorities_used",
    "limits": "limits",
}

TRIAGE_EXAMPLES = {
    "Custom": {},
    "Consumer refund dispute": {
        "triage_case_type": "Consumer dispute",
        "triage_user_role": "Petitioner / Appellant / Complainant / Applicant",
        "triage_forum": "Consumer Commission / Forum",
        "triage_facts_input": (
            "The complainant says an online marketplace delivered a defective electronic product "
            "and refused to refund the amount despite a timely complaint, return pickup, and repeated support requests."
        ),
        "triage_relief_input": "Refund of the purchase price, compensation, and litigation costs.",
        "triage_evidence_input": (
            "Order receipt, delivery proof, complaint emails, return pickup confirmation, "
            "product photos, and customer support chat transcripts."
        ),
        "triage_opponent_input": (
            "The seller and platform say the product was damaged after delivery and that refund conditions were not met."
        ),
        "triage_narrative_input": "",
    },
    "University result dispute": {
        "triage_case_type": "University / examination dispute",
        "triage_user_role": "Petitioner / Appellant / Complainant / Applicant",
        "triage_forum": "High Court",
        "triage_facts_input": (
            "The student challenges cancellation of examination results for alleged unfair means. "
            "No incriminating material was recovered from her possession, the inquiry was perfunctory, "
            "and she says she did not get a meaningful chance to explain her position."
        ),
        "triage_relief_input": "Restoration of results and permission to continue the course without academic prejudice.",
        "triage_evidence_input": "Admit card, invigilator notes, inquiry notice, hearing emails, and university regulations.",
        "triage_opponent_input": (
            "The university says invigilator reports and surrounding circumstances justified disciplinary action."
        ),
        "triage_narrative_input": "",
    },
    "Tax dispute": {
        "triage_case_type": "Tax dispute",
        "triage_user_role": "Petitioner / Appellant / Complainant / Applicant",
        "triage_forum": "Tribunal",
        "triage_facts_input": (
            "The assessee challenges disallowance of expenditure claimed as revenue in nature and says "
            "the payments were wholly for business purposes."
        ),
        "triage_relief_input": "Deletion of the addition and the resulting tax demand.",
        "triage_evidence_input": "Invoices, ledger extracts, bank statements, tax computation, and prior assessment records.",
        "triage_opponent_input": "The revenue argues that the expenditure created an enduring benefit and should be treated as capital.",
        "triage_narrative_input": "",
    },
}


def get_api_url() -> str:
    return os.getenv("STREAMLIT_API_URL", "http://127.0.0.1:8000").rstrip("/")


def health_check(api_url: str) -> dict:
    with httpx.Client(timeout=10) as client:
        response = client.get(f"{api_url}/health")
        response.raise_for_status()
        return response.json()


def call_prediction_api(api_url: str, payload: dict) -> dict:
    with httpx.Client(timeout=180) as client:
        response = client.post(f"{api_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()


def call_question_api(api_url: str, payload: dict) -> dict:
    with httpx.Client(timeout=240) as client:
        response = client.post(f"{api_url}/ask", json=payload)
        response.raise_for_status()
        return response.json()


def upload_session_document(api_url: str, session_id: str, uploaded_file) -> dict:
    with httpx.Client(timeout=240) as client:
        response = client.post(
            f"{api_url}/session-documents",
            json={
                "session_id": session_id,
                "filename": uploaded_file.name,
                "content_type": uploaded_file.type or "application/octet-stream",
                "file_base64": base64.b64encode(uploaded_file.getvalue()).decode("utf-8"),
            },
        )
        response.raise_for_status()
        return response.json()


def clear_session_document(api_url: str, session_id: str) -> dict:
    with httpx.Client(timeout=45) as client:
        response = client.delete(f"{api_url}/session-documents/{session_id}")
        response.raise_for_status()
        return response.json()


def handle_document_upload(api_url: str, uploaded_file) -> None:
    if uploaded_file is None:
        st.error("Choose a document first.")
        return
    try:
        file_bytes = uploaded_file.getvalue()
        st.session_state.uploaded_document_info = upload_session_document(
            api_url,
            st.session_state.ui_session_id,
            uploaded_file,
        )
        st.session_state.uploaded_document_info["file_size_bytes"] = len(file_bytes)
        st.session_state.flash_notice = (
            f"Attached {st.session_state.uploaded_document_info['filename']} to this conversation."
        )
        st.rerun()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        st.error(f"Document upload failed: {detail}")
    except Exception as exc:
        st.error(f"Document upload failed: {exc}")


def handle_document_clear(api_url: str) -> None:
    try:
        clear_session_document(api_url, st.session_state.ui_session_id)
        st.session_state.uploaded_document_info = None
        st.session_state.flash_notice = "Removed the attached document from this conversation."
        st.rerun()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        st.error(f"Could not remove the uploaded document: {detail}")
    except Exception as exc:
        st.error(f"Could not remove the uploaded document: {exc}")


def load_case_detail(api_url: str, case_id: str) -> dict:
    with httpx.Client(timeout=45) as client:
        response = client.get(f"{api_url}/cases/{case_id}")
        response.raise_for_status()
        return response.json()


def render_metric(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        (
            '<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-caption">{caption}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_probability_table(probabilities: dict[str, float]) -> None:
    df = pd.DataFrame(
        [{"Outcome": outcome, "Probability %": value} for outcome, value in probabilities.items()]
    ).sort_values("Probability %", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_notice_box(title: str, items: list[str]) -> None:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return
    bullets = "".join(f"<li>{item}</li>" for item in cleaned)
    st.markdown(
        (
            '<div class="notice-box">'
            f'<div class="notice-title">{title}</div>'
            f"<ul>{bullets}</ul>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_overview_card(title: str, items: list[str], empty_text: str) -> None:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if cleaned:
        body = "".join(f"<p>{item}</p>" for item in cleaned)
    else:
        body = f"<p>{empty_text}</p>"
    st.markdown(
        (
            '<div class="overview-card">'
            f'<div class="overview-title">{title}</div>'
            f"{body}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def make_status_pills(items: list[tuple[str, str]]) -> str:
    return "".join(
        f'<span class="status-pill {tone}">{html.escape(label)}</span>' for label, tone in items
    )


def human_file_size(size_bytes: int | None) -> str:
    if not size_bytes:
        return "Unknown size"
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{int(size_bytes)} B"


@st.cache_data(ttl=20, show_spinner=False)
def get_health_snapshot(api_url: str) -> dict | None:
    try:
        with httpx.Client(timeout=3) as client:
            response = client.get(f"{api_url}/health")
            response.raise_for_status()
            return response.json()
    except Exception:
        return None


def extract_case_ids(text: str) -> list[str]:
    if not text:
        return []
    candidates = re.findall(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+){2,}\b", text)
    filtered = [
        candidate
        for candidate in candidates
        if re.search(r"_(?:19|20)\d{2}(?:_|$)", candidate) or re.search(r"_\d{3,}(?:_|$)", candidate)
    ]
    return list(dict.fromkeys(filtered))


def parse_answer_card(answer_text: str, payload: dict | None = None) -> dict:
    cleaned_answer = (answer_text or "").strip()
    body = re.sub(r"(?im)^sources?\s*:\s*.*$", "", cleaned_answer).strip()

    source_ids: list[str] = []
    if payload:
        workspace = payload.get("workspace") or {}
        source_ids.extend(workspace.get("current_scope_case_ids") or [])
        if not source_ids:
            source_ids.extend(
                [item.get("case_id") for item in (payload.get("supporting_cases") or []) if item.get("case_id")]
            )
        source_mode = payload.get("source_mode")
        if (
            not source_ids
            and source_mode
            in {
                "document_only",
                "document_plus_case",
                "document_plus_reference_law",
                "document_plus_reference_law_plus_case",
            }
            and st.session_state.uploaded_document_info
        ):
            source_ids.append(st.session_state.uploaded_document_info["filename"])
    source_ids.extend(extract_case_ids(cleaned_answer))
    source_ids = list(dict.fromkeys([item for item in source_ids if item]))[:8]

    advisories = list((payload or {}).get("advisories") or [])
    warning_phrases = [
        "not enough evidence",
        "insufficient evidence",
        "cannot answer reliably",
        "could not answer this reliably",
        "off-domain",
        "weak evidence",
        "limited support",
    ]
    if not advisories:
        lowered = cleaned_answer.lower()
        if any(phrase in lowered for phrase in warning_phrases):
            advisories = ["The retrieved support appears limited, mixed, or outside the strongest domain match."]

    return {
        "body": body or cleaned_answer,
        "source_ids": source_ids,
        "advisories": advisories,
        "retrieval_confidence": (payload or {}).get("retrieval_confidence"),
        "evidence_strength": (payload or {}).get("evidence_strength"),
        "answer_confidence": (payload or {}).get("answer_confidence"),
    }


def build_evidence_signal(
    advisories: list[str],
    source_count: int,
    posture: str | None = None,
    *,
    retrieval_confidence: str | None = None,
    evidence_strength: str | None = None,
    answer_confidence: str | None = None,
) -> str | None:
    if retrieval_confidence or evidence_strength or answer_confidence:
        parts: list[str] = []
        if retrieval_confidence:
            parts.append(f"Retrieval: {retrieval_confidence.capitalize()}")
        if evidence_strength:
            label = "Evidence"
            value = evidence_strength.replace("_", " ").capitalize()
            parts.append(f"{label}: {value}")
        if answer_confidence:
            parts.append(f"Answer: {answer_confidence.capitalize()}")
        if parts:
            return " · ".join(parts)
    lowered = " ".join(item.lower() for item in (advisories or []))
    posture_lower = (posture or "").lower()
    if any(
        phrase in lowered
        for phrase in [
            "mixed",
            "weak evidence",
            "limited support",
            "off-domain",
            "cannot answer reliably",
            "not enough evidence",
        ]
    ):
        return "Evidence strength: Mixed"
    if "caution" in posture_lower:
        return "Confidence: Moderate"
    if source_count >= 3:
        return "Evidence strength: Supported"
    if source_count >= 1:
        return "Confidence: Moderate"
    return None


def build_review_signal(result: dict) -> str:
    retrieval_confidence = normalize_whitespace(result.get("retrieval_confidence"))
    evidence_strength = normalize_whitespace(result.get("evidence_strength"))
    answer_confidence = normalize_whitespace(result.get("answer_confidence"))
    if retrieval_confidence or evidence_strength or answer_confidence:
        parts: list[str] = []
        if retrieval_confidence:
            parts.append(f"Retrieval: {retrieval_confidence.capitalize()}")
        if evidence_strength:
            parts.append(f"Evidence: {evidence_strength.replace('_', ' ').capitalize()}")
        if answer_confidence:
            parts.append(f"Review confidence: {answer_confidence.capitalize()}")
        if parts:
            return " · ".join(parts)
    advisories = result.get("advisories") or []
    lowered = " ".join(item.lower() for item in advisories)
    if any(term in lowered for term in ["mixed", "distinguish", "manual review", "caution"]):
        return "Evidence strength: Mixed"
    confidence = float(result.get("confidence_score") or 0.0)
    if confidence >= 80:
        return "Confidence: High"
    if confidence >= 60:
        return "Confidence: Moderate"
    return "Confidence: Preliminary"


def render_source_chips(source_ids: list[str], label: str = "Sources used") -> None:
    st.markdown(f'<div class="message-section-title">{html.escape(label)}</div>', unsafe_allow_html=True)
    if not source_ids:
        st.caption("No source IDs were separated out for this response.")
        return
    chips = "".join(
        f'<span class="source-chip"><span class="source-icon">+</span>{html.escape(case_id)}</span>'
        for case_id in source_ids
    )
    st.markdown(chips, unsafe_allow_html=True)


def text_to_html_blocks(text: str) -> str:
    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", (text or "").strip()) if segment.strip()]
    if not paragraphs:
        return ""
    return "".join(
        f"<p>{html.escape(paragraph).replace(chr(10), '<br>')}</p>" for paragraph in paragraphs
    )


def lines_to_html(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    if all(line.startswith("- ") for line in lines):
        items = "".join(f"<li>{html.escape(line[2:])}</li>" for line in lines)
        return f"<ul>{items}</ul>"
    return "".join(f"<p>{html.escape(line)}</p>" for line in lines)


def source_chips_html(source_ids: list[str], label: str = "Sources used") -> str:
    if not source_ids:
        return ""
    chips = "".join(
        f'<span class="source-chip"><span class="source-icon">+</span>{html.escape(case_id)}</span>'
        for case_id in source_ids
    )
    return (
        f'<div class="message-divider"></div>'
        f'<div class="message-section-title">{html.escape(label)}</div>'
        f"{chips}"
    )


def render_user_message(content: str) -> None:
    st.markdown(
        (
            '<div class="message-card user">'
            '<div class="message-label">Question</div>'
            f'<div class="message-copy">{text_to_html_blocks(content) or html.escape(content)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def format_relevance_score(value: float | int | str | None, item: dict | None = None) -> str:
    try:
        final_score = float(value or 0.0)
    except Exception:
        final_score = 0.0
    base_score = 0.0
    lexical_score = 0.0
    fit_bonus = 0.0
    if isinstance(item, dict):
        try:
            base_score = float(item.get("base_similarity") or 0.0)
        except Exception:
            base_score = 0.0
        try:
            lexical_score = float(item.get("lexical_similarity") or 0.0)
        except Exception:
            lexical_score = 0.0
        fit_band = clean_text(item.get("fit_band")) or ""
        fit_bonus = {"High": 0.14, "Moderate": 0.08, "Low": 0.03}.get(fit_band.title(), 0.0)
    blended = (final_score * 0.3) + (base_score * 0.45) + (lexical_score * 0.25) + fit_bonus
    blended = min(max(blended, 0.18), 0.96)
    return f"{round(blended * 100):d}%"


def humanize_case_fit_note(note: str | None) -> str:
    cleaned = clean_text(note) or ""
    if not cleaned:
        return ""
    replacements = {
        "Subtype match:": "The issue type is close:",
        "Only broad domain match; issue subtype is not clearly aligned.": "The case is from the same legal area, but the factual issue is not perfectly aligned.",
        "Different relief context": "The relief sought appears different.",
        "Same issue": "The issue appears close.",
        "Same remedy": "The remedy appears close.",
        "Same procedural posture": "The procedural posture appears close.",
        "Distinguishable on evidence": "The evidence pattern may be different.",
    }
    for source, target in replacements.items():
        cleaned = re.sub(re.escape(source), target, cleaned, flags=re.I)
    cleaned = re.sub(r"\s+\|\s+.*$", "", cleaned)
    return cleaned


def build_case_help_text(item: dict) -> str:
    fit_note = humanize_case_fit_note(item.get("fit_note") or item.get("retrieval_note"))
    excerpt = clean_text(item.get("excerpt") or item.get("summary") or "") or "No short passage is available."
    excerpt = shorten_text(excerpt, 220)
    if fit_note:
        return f"{fit_note} The retrieved passage indicates: {excerpt}"
    return f"The retrieved passage indicates: {excerpt}"


def build_case_similarity_markdown(item: dict, *, helpful: bool) -> str:
    fit_note = humanize_case_fit_note(item.get("fit_note") or item.get("retrieval_note"))
    excerpt = clean_text(item.get("excerpt") or item.get("summary") or "") or "No short passage is available."
    excerpt = shorten_text(excerpt, 220)
    outcome = item.get("label_name") or "Unknown"
    if helpful:
        why_similar = fit_note or "The issue, remedy, or factual pattern appears reasonably close on the current shortlist."
        application = "This may help your side if your facts and timeline stay close to what the forum accepted in that matter."
    else:
        why_similar = fit_note or "This authority may represent a weaker or more limited outcome on similar facts."
        application = "This could hurt your case if the other side can frame your facts closer to this authority."
    return (
        f"**Why it is relevant**\n\n{why_similar}\n\n"
        f"**What the forum appears to have decided**\n\n"
        f"The retrieved authority is tagged as **{outcome}** in the current dataset.\n\n"
        f"**How it may apply here**\n\n{application}\n\n"
        f"**Retrieved passage**\n\n{excerpt}"
    )


def render_assistant_message(content: str, payload: dict | None = None) -> None:
    parsed = parse_answer_card(content, payload)
    meta_items: list[tuple[str, str]] = []
    source_mode = (payload or {}).get("source_mode")
    if source_mode == "document_only":
        meta_items.append(("Document only", "info"))
    elif source_mode == "reference_law_only":
        meta_items.append(("Official law only", "info"))
    elif source_mode == "reference_law_plus_case":
        meta_items.append(("Law + cases", "info"))
    elif source_mode == "document_plus_reference_law":
        meta_items.append(("Document + law", "info"))
    elif source_mode == "document_plus_reference_law_plus_case":
        meta_items.append(("Document + law + cases", "info"))
    elif source_mode == "case_corpus_only":
        meta_items.append(("Case corpus only", "neutral"))
    elif source_mode == "document_plus_case":
        meta_items.append(("Document + cases", "info"))
    if parsed["source_ids"]:
        meta_items.append((f"{len(parsed['source_ids'])} sources", "neutral"))
    st.markdown('<div class="message-label">Answer</div>', unsafe_allow_html=True)
    if meta_items:
        st.markdown(
            f'<div class="message-meta">{make_status_pills(meta_items)}</div>',
            unsafe_allow_html=True,
        )
    st.markdown(parsed["body"] or "No answer text returned.")
    signal_line = build_evidence_signal(
        parsed["advisories"],
        len(parsed["source_ids"]),
        (payload or {}).get("prediction_posture"),
        retrieval_confidence=parsed.get("retrieval_confidence"),
        evidence_strength=parsed.get("evidence_strength"),
        answer_confidence=parsed.get("answer_confidence"),
    )
    if signal_line:
        st.markdown(f'<div class="signal-line"><strong>{html.escape(signal_line)}</strong></div>', unsafe_allow_html=True)
    if parsed["source_ids"]:
        render_source_chips(parsed["source_ids"])


def render_loading_message(text: str) -> None:
    st.markdown(
        (
            '<div class="message-card assistant">'
            '<div class="message-label">Answer</div>'
            f'<div class="loading-message"><span class="loading-dot"></span><span>{html.escape(text)}</span></div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def render_chat_empty_state() -> None:
    return


def _legacy_render_attachment_popover(api_url: str, key_prefix: str) -> None:
    popover_factory = getattr(st, "popover", None)
    context = popover_factory("📎") if callable(popover_factory) else st.expander("📎", expanded=False)
    with context:
        if st.session_state.uploaded_document_info:
            doc_info = st.session_state.uploaded_document_info
            st.markdown(f"**{doc_info['filename']}**")
            st.caption(
                f"{doc_info.get('chunk_count', 0)} chunks · {doc_info.get('word_count', 0)} words · document-only answers"
            )
            replacement_file = st.file_uploader(
                "Replace uploaded document",
                type=["txt", "md", "docx", "pdf"],
                key=f"{key_prefix}_replacement_document_popover",
                label_visibility="collapsed",
            )
            action_left, action_right = st.columns(2)
            with action_left:
                if st.button(
                    "Replace",
                    key=f"{key_prefix}_replace_document_popover",
                    use_container_width=True,
                    disabled=replacement_file is None,
                ):
                    handle_document_upload(api_url, replacement_file)
                    st.rerun()
            with action_right:
                if st.button(
                    "Remove",
                    key=f"{key_prefix}_remove_document_popover",
                    use_container_width=True,
                ):
                    handle_document_clear(api_url)
                    st.rerun()
        else:
            uploaded_file = st.file_uploader(
                "Attach legal document",
                type=["txt", "md", "docx", "pdf"],
                key=f"{key_prefix}_uploaded_document_popover",
                label_visibility="collapsed",
            )
            st.caption("Attach one document if you want a fast document-only answer.")
            if st.button(
                "Attach",
                key=f"{key_prefix}_attach_document_popover",
                use_container_width=True,
                disabled=uploaded_file is None,
            ):
                handle_document_upload(api_url, uploaded_file)
                st.rerun()


def render_document_sidebar(api_url: str, key_prefix: str) -> None:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Attach legal document</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Supported formats: PDF, DOCX, TXT. Uploaded material stays in the current session.</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Attach legal document",
        type=["txt", "md", "docx", "pdf"],
        key=f"{key_prefix}_uploaded_document",
        label_visibility="collapsed",
    )
    if st.session_state.uploaded_document_info:
        doc_info = st.session_state.uploaded_document_info
        st.markdown(
            make_status_pills(
                [
                    ("Document active", "good"),
                    (f"{doc_info.get('chunk_count', 0)} chunks", "neutral"),
                    (f"{doc_info.get('word_count', 0)} words", "neutral"),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                f"**{doc_info['filename']}**  \n"
                f"{human_file_size(doc_info.get('file_size_bytes'))} | "
                f"{doc_info.get('chunk_count', 0)} parsed chunks"
            )
        )
        st.caption("Answers can use both the uploaded document and the case-law corpus.")
        action_left, action_right = st.columns(2)
        with action_left:
            if st.button(
                "Replace document",
                key=f"{key_prefix}_replace_doc",
                use_container_width=True,
                disabled=uploaded_file is None,
            ):
                handle_document_upload(api_url, uploaded_file)
        with action_right:
            if st.button("Remove document", key=f"{key_prefix}_remove_doc", use_container_width=True):
                handle_document_clear(api_url)
    else:
        st.caption("Upload a petition, notice, order, complaint, or judgment for document-aware analysis.")
        if st.button(
            "Attach document",
            key=f"{key_prefix}_attach_doc",
            use_container_width=True,
            disabled=uploaded_file is None,
        ):
            handle_document_upload(api_url, uploaded_file)
    st.markdown("</div>", unsafe_allow_html=True)


def render_document_summary_card(api_url: str, key_prefix: str) -> None:
    doc_info = st.session_state.uploaded_document_info
    if not doc_info:
        return
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Uploaded document</div>', unsafe_allow_html=True)
    st.markdown(
        make_status_pills(
            [
                ("Document active", "good"),
                (f"{doc_info.get('chunk_count', 0)} chunks", "neutral"),
                (f"{doc_info.get('word_count', 0)} words", "neutral"),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            f"**{doc_info['filename']}**  \n"
            f"{human_file_size(doc_info.get('file_size_bytes'))} | "
            f"{doc_info.get('chunk_count', 0)} chunks | {doc_info.get('word_count', 0)} words"
        )
    )
    st.caption("Q/A answers can use this uploaded document together with official law materials and related authorities.")
    preview = (doc_info.get("preview_text") or "").strip()
    if preview:
        st.markdown(f'<div class="doc-summary">{html.escape(preview)}</div>', unsafe_allow_html=True)
    replacement_file = st.file_uploader(
        "Replace uploaded document",
        type=["txt", "md", "docx", "pdf"],
        key=f"{key_prefix}_replacement_document",
        label_visibility="collapsed",
    )
    action_left, action_right = st.columns(2)
    with action_left:
        if st.button(
            "Replace",
            key=f"{key_prefix}_replace_document",
            use_container_width=True,
            disabled=replacement_file is None,
        ):
            handle_document_upload(api_url, replacement_file)
    with action_right:
        if st.button(
            "Remove",
            key=f"{key_prefix}_remove_document",
            use_container_width=True,
        ):
            handle_document_clear(api_url)
    st.markdown("</div>", unsafe_allow_html=True)


def render_document_inline_strip(api_url: str, key_prefix: str) -> None:
    if st.session_state.uploaded_document_info:
        return

    st.markdown(
        '<div class="composer-note">Optional: attach one legal document if you want answers grounded only in that file.</div>',
        unsafe_allow_html=True,
    )
    uploader_col, action_col = st.columns([5.2, 1.0])
    with uploader_col:
        uploaded_file = st.file_uploader(
            "Attach legal document",
            type=["txt", "md", "docx", "pdf"],
            key=f"{key_prefix}_uploaded_document_inline",
            label_visibility="collapsed",
        )
    with action_col:
        if st.button(
            "Attach",
            key=f"{key_prefix}_attach_doc_inline",
            use_container_width=True,
            disabled=uploaded_file is None,
        ):
            handle_document_upload(api_url, uploaded_file)


def _unused_render_attachment_popover_legacy(api_url: str, key_prefix: str) -> None:
    popover_factory = getattr(st, "popover", None)
    context = popover_factory("📎") if callable(popover_factory) else st.expander("📎", expanded=False)
    with context:
        if st.session_state.uploaded_document_info:
            doc_info = st.session_state.uploaded_document_info
            st.markdown(f"**{doc_info['filename']}**")
            st.caption(
                f"{doc_info.get('chunk_count', 0)} chunks | {doc_info.get('word_count', 0)} words | document-only answers"
            )
            replacement_file = st.file_uploader(
                "Replace uploaded document",
                type=["txt", "md", "docx", "pdf"],
                key=f"{key_prefix}_replacement_document_popover_clean",
                label_visibility="collapsed",
            )
            action_left, action_right = st.columns(2)
            with action_left:
                if st.button(
                    "Replace",
                    key=f"{key_prefix}_replace_document_popover_clean",
                    use_container_width=True,
                    disabled=replacement_file is None,
                ):
                    handle_document_upload(api_url, replacement_file)
                    st.rerun()
            with action_right:
                if st.button(
                    "Remove",
                    key=f"{key_prefix}_remove_document_popover_clean",
                    use_container_width=True,
                ):
                    handle_document_clear(api_url)
                    st.rerun()
        else:
            uploaded_file = st.file_uploader(
                "Attach legal document",
                type=["txt", "md", "docx", "pdf"],
                key=f"{key_prefix}_uploaded_document_popover_clean",
                label_visibility="collapsed",
            )
            st.caption("Attach one document for fast document-only answers.")
            if st.button(
                "Attach",
                key=f"{key_prefix}_attach_document_popover_clean",
                use_container_width=True,
                disabled=uploaded_file is None,
            ):
                handle_document_upload(api_url, uploaded_file)
                st.rerun()

def render_attachment_popover(api_url: str, key_prefix: str) -> None:
    popover_factory = getattr(st, "popover", None)
    context = popover_factory("+") if callable(popover_factory) else st.expander("+", expanded=False)
    with context:
        if st.session_state.uploaded_document_info:
            doc_info = st.session_state.uploaded_document_info
            st.markdown(f"**{doc_info['filename']}**")
            st.caption(
                f"{doc_info.get('chunk_count', 0)} chunks | {doc_info.get('word_count', 0)} words | document-only answers"
            )
            replacement_file = st.file_uploader(
                "Replace uploaded document",
                type=["txt", "md", "docx", "pdf"],
                key=f"{key_prefix}_replacement_document_popover_plus",
                label_visibility="collapsed",
            )
            action_left, action_right = st.columns(2)
            with action_left:
                if st.button(
                    "Replace",
                    key=f"{key_prefix}_replace_document_popover_plus",
                    use_container_width=True,
                    disabled=replacement_file is None,
                ):
                    handle_document_upload(api_url, replacement_file)
                    st.rerun()
            with action_right:
                if st.button(
                    "Remove",
                    key=f"{key_prefix}_remove_document_popover_plus",
                    use_container_width=True,
                ):
                    handle_document_clear(api_url)
                    st.rerun()
        else:
            uploaded_file = st.file_uploader(
                "Attach legal document",
                type=["txt", "md", "docx", "pdf"],
                key=f"{key_prefix}_uploaded_document_popover_plus",
                label_visibility="collapsed",
            )
            st.caption("Attach one document for fast document-only answers.")
            if st.button(
                "Attach",
                key=f"{key_prefix}_attach_document_popover_plus",
                use_container_width=True,
                disabled=uploaded_file is None,
            ):
                handle_document_upload(api_url, uploaded_file)
                st.rerun()


def render_quick_tips(title: str, items: list[str]) -> None:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    bullets = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    st.markdown(f'<ul class="helper-list">{bullets}</ul>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def init_state() -> None:
    configured_api_url = get_api_url()
    defaults = {
        "analysis_result": None,
        "qa_result": None,
        "case_details": {},
        "chat_messages": [],
        "flash_notice": None,
        "pending_chat_updates": None,
        "ui_session_id": str(uuid.uuid4()),
        "uploaded_document_info": None,
        "api_url_input": configured_api_url,
        "ui_top_k": 3,
        "ui_include_explanation": True,
        "qa_draft_input": "",
        "qa_submit_error": None,
        "reset_qa_draft": False,
        "chat_scope": "Search the full case library",
        "chat_source_mode": "Document + legal sources",
        "chat_retrieval_profile": "Fast answer",
        "chat_case_type": "Not specified yet",
        "chat_user_role": "Not specified yet",
        "chat_forum": "Not specified yet",
        "chat_context_note": "",
        "selected_example": "Custom",
        "triage_case_type": "Not specified yet",
        "triage_user_role": "Not specified yet",
        "triage_forum": "Not specified yet",
        "triage_facts_input": "",
        "triage_relief_input": "",
        "triage_evidence_input": "",
        "triage_opponent_input": "",
        "triage_narrative_input": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if os.getenv("STREAMLIT_API_URL"):
        st.session_state.api_url_input = configured_api_url


def apply_example() -> None:
    selected = st.session_state.selected_example
    example = TRIAGE_EXAMPLES.get(selected, {})
    st.session_state.triage_case_type = example.get("triage_case_type", "Not specified yet")
    st.session_state.triage_user_role = example.get("triage_user_role", "Not specified yet")
    st.session_state.triage_forum = example.get("triage_forum", "Not specified yet")
    st.session_state.triage_facts_input = example.get("triage_facts_input", "")
    st.session_state.triage_relief_input = example.get("triage_relief_input", "")
    st.session_state.triage_evidence_input = example.get("triage_evidence_input", "")
    st.session_state.triage_opponent_input = example.get("triage_opponent_input", "")
    st.session_state.triage_narrative_input = example.get("triage_narrative_input", "")


def clean_text(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def build_triage_payload(top_k: int, include_explanation: bool) -> dict:
    return {
        "session_id": st.session_state.ui_session_id,
        "case_type": clean_text(st.session_state.triage_case_type)
        if st.session_state.triage_case_type != "Not specified yet"
        else None,
        "user_role": clean_text(st.session_state.triage_user_role)
        if st.session_state.triage_user_role != "Not specified yet"
        else None,
        "forum": clean_text(st.session_state.triage_forum)
        if st.session_state.triage_forum != "Not specified yet"
        else None,
        "facts": clean_text(st.session_state.triage_facts_input),
        "relief_sought": clean_text(st.session_state.triage_relief_input),
        "evidence_summary": clean_text(st.session_state.triage_evidence_input),
        "opponent_arguments": clean_text(st.session_state.triage_opponent_input),
        "case_text": clean_text(st.session_state.triage_narrative_input),
        "top_k": top_k,
        "include_explanation": include_explanation,
    }


def has_substantive_case_input() -> bool:
    if st.session_state.uploaded_document_info:
        return True
    payload = build_triage_payload(top_k=3, include_explanation=True)
    combined = " ".join(
        [
            payload.get("facts") or "",
            payload.get("relief_sought") or "",
            payload.get("evidence_summary") or "",
            payload.get("opponent_arguments") or "",
            payload.get("case_text") or "",
        ]
    ).strip()
    return len(combined) >= 30


def get_scope_case_ids(source_label: str) -> list[str]:
    mapping = {
        "triage": st.session_state.analysis_result,
        "chat": st.session_state.qa_result,
    }
    result = mapping.get(source_label) or {}
    workspace = result.get("workspace") or {}
    return list(dict.fromkeys(workspace.get("current_scope_case_ids") or []))[:5]


def resolve_chat_source_mode() -> str:
    selected_label = clean_text(st.session_state.get("chat_source_mode") or "")
    has_document = bool(st.session_state.uploaded_document_info)
    if selected_label == "Uploaded document only" and has_document:
        return "document_only"
    if selected_label == "Case corpus only":
        return "case_corpus_only"
    if has_document:
        return "document_plus_case"
    return "document_plus_case"


def build_question_payload(question: str, top_k: int) -> dict:
    selected_source_mode = resolve_chat_source_mode()
    effective_top_k = 1 if selected_source_mode == "document_only" else top_k
    retrieval_profile = (
        "deep" if st.session_state.chat_retrieval_profile == "Deep analysis" else "fast"
    )
    scope = "corpus"
    scope_case_ids: list[str] = []

    if selected_source_mode in {"case_corpus_only", "document_plus_case"}:
        if st.session_state.chat_scope == "Use the evidence already in this chat":
            scope = "current_result"
            scope_case_ids = get_scope_case_ids("chat")
        elif st.session_state.chat_scope == "Use latest review cases":
            scope = "current_result"
            scope_case_ids = get_scope_case_ids("triage")

    return {
        "session_id": st.session_state.ui_session_id,
        "question": clean_text(question),
        "case_type": clean_text(st.session_state.chat_case_type)
        if st.session_state.chat_case_type != "Not specified yet"
        else None,
        "user_role": clean_text(st.session_state.chat_user_role)
        if st.session_state.chat_user_role != "Not specified yet"
        else None,
        "forum": clean_text(st.session_state.chat_forum)
        if st.session_state.chat_forum != "Not specified yet"
        else None,
        "context_note": clean_text(st.session_state.chat_context_note),
        "scope": scope,
        "source_mode": selected_source_mode,
        "retrieval_profile": retrieval_profile,
        "scope_case_ids": scope_case_ids,
        "chat_history": st.session_state.chat_messages[-8:],
        "top_k": effective_top_k,
    }


def parse_chat_submission(submission) -> tuple[str, list]:
    if submission is None:
        return "", []
    if isinstance(submission, str):
        return submission, []
    text = getattr(submission, "text", "")
    files = getattr(submission, "files", None)
    if text is None and isinstance(submission, dict):
        text = submission.get("text", "")
    if files is None and isinstance(submission, dict):
        files = submission.get("files", [])
    return text or "", list(files or [])


def attach_document_from_submission(api_url: str, uploaded_file) -> str | None:
    try:
        file_bytes = uploaded_file.getvalue()
        st.session_state.uploaded_document_info = upload_session_document(
            api_url,
            st.session_state.ui_session_id,
            uploaded_file,
        )
        st.session_state.uploaded_document_info["file_size_bytes"] = len(file_bytes)
        return None
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        return f"Document upload failed: {detail}"
    except Exception as exc:
        return f"Document upload failed: {exc}"


def validate_chat_turn(question: str) -> str | None:
    trimmed_prompt = question.strip()
    if st.session_state.uploaded_document_info:
        min_length = 4
    else:
        min_length = 8 if st.session_state.chat_messages else 12
    if len(trimmed_prompt) < min_length:
        return "Please ask a fuller legal question so the system has enough context to retrieve useful authorities."

    using_case_corpus = not st.session_state.uploaded_document_info
    if (
        using_case_corpus
        and st.session_state.chat_scope == "Use the evidence already in this chat"
        and not get_scope_case_ids("chat")
    ):
        return "Ask one full-library question first, or switch the search mode back to the full case library."
    if (
        using_case_corpus
        and st.session_state.chat_scope == "Use latest review cases"
        and not get_scope_case_ids("triage")
    ):
        return "Run case review first if you want to ask only on the latest reviewed cases."
    return None


def build_chat_recovery_message(*, document_attached: bool) -> str:
    if document_attached:
        return (
            "Your document is attached. Ask a fuller question, or try one of these:\n\n"
            "- Summarize this document.\n"
            "- Explain the key issue in simple language.\n"
            "- Point me to the exact section, clause, or paragraph."
        )
    return (
        "Ask a fuller legal question so retrieval can stay grounded. Good examples are:\n\n"
        "- Compare accepted versus rejected cases on this issue.\n"
        "- What facts usually strengthen this claim?\n"
        "- Answer only from the strongest retrieved authorities."
    )


def render_backend_status(api_url: str) -> None:
    if st.button("Check backend", use_container_width=True):
        try:
            status = health_check(api_url)
            st.success("Backend reachable.")
            if status["retrieval_ready"]:
                st.caption(f"Case similarity store ready: {status['retrieval_record_count']} cases")
            else:
                st.warning("Case similarity store is not ready.")
            if status.get("qa_retrieval_ready"):
                st.caption(f"RAG store ready: {status['qa_retrieval_record_count']} chunks")
            else:
                st.warning("RAG store is not ready.")
            if status.get("embedding_model_name"):
                st.caption(f"Shared retrieval embedding: {status['embedding_model_name']}")
            if status.get("embedding_space_status"):
                st.caption(status["embedding_space_status"])
            if status.get("llm_ready"):
                st.caption(status.get("llm_status_message") or "Local LLM reachable.")
            else:
                st.warning(status.get("llm_status_message") or "Local LLM is not reachable.")
        except Exception as exc:
            st.error(f"Backend unavailable: {exc}")


def render_case_detail(case_id: str, api_url: str, key_prefix: str) -> None:
    if st.button("Open full judgment", key=f"{key_prefix}_open_{case_id}", use_container_width=True):
        try:
            detail = load_case_detail(api_url, case_id)
            st.session_state.case_details[case_id] = detail
        except httpx.HTTPStatusError as exc:
            detail_message = exc.response.text
            try:
                detail_message = exc.response.json().get("detail", detail_message)
            except Exception:
                pass
            st.error(f"Could not load the full judgment: {detail_message}")
        except Exception as exc:
            st.error(f"Could not load the full judgment: {exc}")

    detail = st.session_state.case_details.get(case_id)
    if not detail:
        st.caption("Open the full judgment when you want to verify the reasoning in detail.")
        return

    meta_left, meta_right = st.columns(2)
    meta_left.markdown(f"**Word count**  \n{detail['word_count']}")
    meta_right.markdown(
        f"**Outcome label**  \n{detail.get('label_name') or detail.get('label') or 'Unknown'}"
    )
    st.text_area(
        "Full judgment text",
        value=detail["full_text"],
        height=320,
        disabled=True,
        key=f"{key_prefix}_text_{case_id}",
    )


def render_case_group(title: str, cases: list[dict], api_url: str, key_prefix: str) -> None:
    st.markdown(f"#### {title}")
    if not cases:
        st.caption("No cases are available in this group yet.")
        return

    helpful_group = "risk" not in title.lower() and "counter" not in title.lower() and "conflict" not in title.lower()

    for index, item in enumerate(cases, start=1):
        label = item.get("title") or item["case_id"]
        with st.expander(f"{index}. {label}", expanded=(index == 1)):
            meta_left, meta_right, meta_third = st.columns([2.2, 1, 1.2])
            meta_left.markdown(f"**Case ID**  \n`{item['case_id']}`")
            meta_right.markdown(f"**Relevance**  \n{format_relevance_score(item.get('similarity'), item)}")
            meta_third.markdown(f"**Outcome**  \n{item.get('label_name') or 'Unknown'}")
            chips = [item.get("court"), item.get("case_type"), item.get("date")]
            if item.get("fit_band"):
                chips.append(f"Fact fit: {str(item['fit_band']).title()}")
            if any(chips):
                st.markdown(
                    "".join(
                        f'<span class="authority-tag">{chip}</span>'
                        for chip in chips
                        if chip
                    ),
                    unsafe_allow_html=True,
                )
            st.markdown(build_case_similarity_markdown(item, helpful=helpful_group))
            render_case_detail(item["case_id"], api_url, key_prefix=f"{key_prefix}_{index}")


def render_similar_cases_only(workspace: dict, api_url: str, key_prefix: str) -> None:
    scope_case_ids = workspace.get("current_scope_case_ids") or []
    if scope_case_ids:
        st.markdown("**Cases currently in scope for follow-up**")
        st.markdown(
            " ".join(f'<span class="workspace-chip">{case_id}</span>' for case_id in scope_case_ids),
            unsafe_allow_html=True,
        )

    tab_labels = ["Best support", "Main risks / distinguishable", "Broader same-domain cases"]
    tab_1, tab_2, tab_3 = st.tabs(tab_labels)
    with tab_1:
        render_case_group(tab_labels[0], workspace.get("supporting_authorities") or [], api_url, f"{key_prefix}_support")
    with tab_2:
        render_case_group(tab_labels[1], workspace.get("conflicting_authorities") or [], api_url, f"{key_prefix}_conflict")
    with tab_3:
        render_case_group(tab_labels[2], workspace.get("mixed_authorities") or [], api_url, f"{key_prefix}_mixed")


def build_case_signal_items(cases: list[dict], *, empty_text: str) -> list[str]:
    if not cases:
        return [empty_text]
    items: list[str] = []
    for item in cases[:2]:
        excerpt = clean_text(item.get("excerpt") or item.get("summary") or "") or "No short excerpt available."
        compact = excerpt[:130].rstrip()
        if len(excerpt) > 130:
            compact += "..."
        fit_note = humanize_case_fit_note(item.get("fit_note") or "")
        line = f"{item.get('case_id')}: {compact}"
        if fit_note:
            line += f" {fit_note}"
        items.append(line)
    return items


def render_workspace(workspace: dict, api_url: str, key_prefix: str) -> None:
    st.markdown('<div class="surface">', unsafe_allow_html=True)
    st.markdown(f"### {workspace['headline']}")
    if workspace.get("authority_rationale"):
        st.caption(workspace["authority_rationale"])
    if workspace.get("workflow") == "triage":
        top_left, top_right = st.columns(2, gap="medium")
        bottom_left, bottom_right = st.columns(2, gap="medium")
        with top_left:
            render_overview_card(
                "Case read",
                workspace.get("issue_outline") or [],
                "No stable issue outline yet.",
            )
        with top_right:
            render_overview_card(
                "Support snapshot",
                build_case_signal_items(
                    workspace.get("supporting_authorities") or [],
                    empty_text="No clear supportive authority has surfaced yet.",
                ),
                "No clear supportive authority has surfaced yet.",
            )
        with bottom_left:
            risk_items = build_case_signal_items(
                (workspace.get("conflicting_authorities") or []) + (workspace.get("mixed_authorities") or []),
                empty_text="No strong counter-authority is surfaced yet, so distinguishable risks must be checked manually.",
            )
            risk_items.extend((workspace.get("evidence_gaps") or [])[:1])
            render_overview_card(
                "Main risks",
                risk_items,
                "No strong counter-authority is surfaced yet.",
            )
        with bottom_right:
            render_overview_card(
                "Next checks",
                workspace.get("next_steps") or [],
                "No follow-up steps suggested yet.",
            )
    else:
        issues_col, gaps_col, steps_col = st.columns(3)
        with issues_col:
            render_overview_card(
                "Key issues",
                workspace.get("issue_outline") or [],
                "No stable issue outline yet.",
            )
        with gaps_col:
            render_overview_card(
                "What is still missing",
                workspace.get("evidence_gaps") or [],
                "No major evidence gaps are flagged right now.",
            )
        with steps_col:
            render_overview_card(
                "Best next steps",
                workspace.get("next_steps") or [],
                "No follow-up steps suggested yet.",
            )

    scope_case_ids = workspace.get("current_scope_case_ids") or []
    if scope_case_ids:
        st.markdown("**Cases currently in scope for follow-up**")
        st.markdown(
            " ".join(
                f'<span class="workspace-chip">{case_id}</span>' for case_id in scope_case_ids
            ),
            unsafe_allow_html=True,
        )

    if workspace.get("workflow") == "triage":
        tab_labels = ["Best support", "Main risks / distinguishable", "Broader same-domain cases"]
    else:
        tab_labels = ["Leading authorities", "Counter-authorities", "Additional authorities"]

    tab_1, tab_2, tab_3 = st.tabs(tab_labels)
    with tab_1:
        render_case_group(tab_labels[0], workspace.get("supporting_authorities") or [], api_url, f"{key_prefix}_support")
    with tab_2:
        render_case_group(tab_labels[1], workspace.get("conflicting_authorities") or [], api_url, f"{key_prefix}_conflict")
    with tab_3:
        render_case_group(tab_labels[2], workspace.get("mixed_authorities") or [], api_url, f"{key_prefix}_mixed")
    st.markdown("</div>", unsafe_allow_html=True)


def render_chat_sources(workspace: dict, api_url: str, key_prefix: str) -> None:
    scope_case_ids = workspace.get("current_scope_case_ids") or []
    if scope_case_ids:
        st.markdown("**Cases currently in scope for follow-up**")
        st.markdown(
            " ".join(f'<span class="workspace-chip">{case_id}</span>' for case_id in scope_case_ids),
            unsafe_allow_html=True,
        )

    tab_1, tab_2, tab_3 = st.tabs(["Top sources", "Counter-sources", "More sources"])
    with tab_1:
        render_case_group(
            "Top sources",
            workspace.get("supporting_authorities") or [],
            api_url,
            f"{key_prefix}_support",
        )
    with tab_2:
        render_case_group(
            "Counter-sources",
            workspace.get("conflicting_authorities") or [],
            api_url,
            f"{key_prefix}_conflict",
        )
    with tab_3:
        render_case_group(
            "More sources",
            workspace.get("mixed_authorities") or [],
            api_url,
            f"{key_prefix}_mixed",
        )


def process_chat_submission(api_url: str, submission, top_k: int, response_slot) -> None:
    question, files = parse_chat_submission(submission)
    trimmed_question = question.strip()

    if files:
        attach_error = attach_document_from_submission(api_url, files[0])
        if attach_error:
            st.session_state.qa_submit_error = attach_error
            return

    if not trimmed_question:
        if files:
            st.session_state.flash_notice = "Document attached to this conversation."
            helper_message = build_chat_recovery_message(document_attached=True)
            with response_slot.container():
                render_assistant_message(
                    helper_message,
                    {
                        "source_mode": resolve_chat_source_mode(),
                        "retrieval_confidence": "low",
                        "evidence_strength": "insufficient",
                        "answer_confidence": "low",
                    },
                )
        return

    validation_error = validate_chat_turn(trimmed_question)
    if validation_error:
        if st.session_state.uploaded_document_info:
            min_length = 4
        else:
            min_length = 8 if st.session_state.chat_messages else 12
        if len(trimmed_question) < min_length:
            helper_message = build_chat_recovery_message(
                document_attached=bool(st.session_state.uploaded_document_info)
            )
            with response_slot.container():
                render_assistant_message(
                    helper_message,
                    {
                        "source_mode": resolve_chat_source_mode(),
                        "retrieval_confidence": "low",
                        "evidence_strength": "insufficient",
                        "answer_confidence": "low",
                    },
                )
        else:
            st.session_state.qa_submit_error = validation_error
        return

    st.session_state.chat_messages.append({"role": "user", "content": trimmed_question})
    with response_slot.container():
        render_loading_message(
            "Reading the uploaded document and drafting the answer..."
            if st.session_state.uploaded_document_info
            else "Searching retrieved authorities and drafting answer..."
        )

    try:
        st.session_state.qa_result = call_question_api(
            api_url,
            build_question_payload(trimmed_question, top_k),
        )
        st.session_state.case_details = {}
        assistant_message = {
            "role": "assistant",
            "content": st.session_state.qa_result["answer"],
            "payload": st.session_state.qa_result,
        }
        st.session_state.chat_messages.append(assistant_message)
        with response_slot.container():
            render_assistant_message(
                assistant_message["content"],
                assistant_message["payload"],
            )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        error_message = f"Chat failed: {detail}"
        st.session_state.chat_messages.append({"role": "assistant", "content": error_message})
        with response_slot.container():
            render_assistant_message(error_message, None)
    except Exception as exc:
        error_message = f"Chat failed: {exc}"
        st.session_state.chat_messages.append({"role": "assistant", "content": error_message})
        with response_slot.container():
            render_assistant_message(error_message, None)


def clear_results() -> None:
    st.session_state.analysis_result = None
    st.session_state.qa_result = None
    st.session_state.chat_messages = []
    st.session_state.case_details = {}


def queue_chat_context_from_triage(use_review_cases: bool = False) -> None:
    pending = dict(st.session_state.pending_chat_updates or {})
    if st.session_state.triage_case_type != "Not specified yet":
        pending["chat_case_type"] = st.session_state.triage_case_type
    if st.session_state.triage_user_role != "Not specified yet":
        pending["chat_user_role"] = st.session_state.triage_user_role
    if st.session_state.triage_forum != "Not specified yet":
        pending["chat_forum"] = st.session_state.triage_forum
    fact_note = clean_text(st.session_state.triage_facts_input)
    if fact_note:
        pending["chat_context_note"] = fact_note
    if use_review_cases:
        pending["chat_scope"] = "Use latest review cases"
    st.session_state.pending_chat_updates = pending


def apply_pending_chat_updates() -> None:
    pending = st.session_state.pending_chat_updates
    if not pending:
        return
    for key, value in pending.items():
        st.session_state[key] = value
    st.session_state.pending_chat_updates = None


init_state()
apply_pending_chat_updates()
api_url = st.session_state.api_url_input
status_snapshot = get_health_snapshot(st.session_state.api_url_input)
status_badges = [
    ("API connected", "good" if status_snapshot else "warn"),
    (
        "Corpus indexed",
        "good"
        if status_snapshot and status_snapshot.get("retrieval_ready") and status_snapshot.get("qa_retrieval_ready")
        else "neutral",
    ),
    ("Document active", "info" if st.session_state.uploaded_document_info else "neutral"),
]
api_status_text = "Connected" if status_snapshot else "Unavailable"
corpus_status_text = (
    "Indexed"
    if status_snapshot and status_snapshot.get("retrieval_ready") and status_snapshot.get("qa_retrieval_ready")
    else "Pending"
)
document_status_text = "Active" if st.session_state.uploaded_document_info else "Inactive"
st.markdown(
    f"""
    <div class="top-header">
        <div class="top-header-grid">
            <div class="header-body">
                <div class="hero-title">Nyaya Case Insight</div>
                <div class="hero-subtitle">
                    Search Indian judgments, review uploaded case documents, and ask grounded questions.
                </div>
            </div>
            <div class="header-status-shell">
                <div class="header-status-grid">
                    <div class="header-status-item">
                        <span class="header-status-label">API</span>
                        <span class="header-status-value">{html.escape(api_status_text)}</span>
                    </div>
                    <div class="header-status-item">
                        <span class="header-status-label">Corpus</span>
                        <span class="header-status-value">{html.escape(corpus_status_text)}</span>
                    </div>
                    <div class="header-status-item">
                        <span class="header-status-label">Document</span>
                        <span class="header-status-value">{html.escape(document_status_text)}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
api_url = st.session_state.api_url_input

if st.session_state.flash_notice:
    st.success(st.session_state.flash_notice)
    st.session_state.flash_notice = None


chat_tab, triage_tab = st.tabs(["Case Q/A", "Case Review"])


with chat_tab:
    qa_main, qa_side = st.columns([3.0, 1.35], gap="large")
    with qa_main:
        top_left, top_mid, top_right = st.columns([2.55, 0.72, 0.88])
        with top_left:
            st.markdown(
                """
                <div class="page-head">
                    <div class="page-title">Case Q/A</div>
                    <div class="page-copy">
                        Work from uploaded facts, retrieved authorities, or both in one grounded legal workspace.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with top_mid:
            if st.button("New chat", key="qa_new_chat", use_container_width=True):
                st.session_state.chat_messages = []
                st.session_state.qa_result = None
                st.session_state.case_details = {}
                st.rerun()
        with top_right:
            if st.button("Clear results", key="qa_clear_results", use_container_width=True):
                clear_results()
                st.rerun()

        messages_container = st.container()
        composer_container = st.container()

        with messages_container:
            for message in st.session_state.chat_messages:
                if message["role"] == "user":
                    render_user_message(message["content"])
                else:
                    payload = message.get("payload") if isinstance(message, dict) else None
                    render_assistant_message(message["content"], payload)

            qa_result = st.session_state.qa_result
            if qa_result and qa_result.get("workspace"):
                show_authorities = st.toggle(
                    "Inspect retrieved authorities",
                    value=False,
                    key="show_chat_authorities",
                )
                if show_authorities:
                    render_chat_sources(qa_result["workspace"], api_url, key_prefix="chat_workspace")

        with composer_container:
            if st.session_state.uploaded_document_info:
                doc_info = st.session_state.uploaded_document_info
                st.markdown(
                    f'<div class="composer-status"><strong>Attached:</strong> {html.escape(doc_info["filename"])} &middot; document-only mode</div>',
                    unsafe_allow_html=True,
                )
            chat_submission = st.chat_input(
                placeholder="Ask a legal question...",
                key="qa_chat_input",
                accept_file=True,
                file_type=["txt", "md", "docx", "pdf"],
            )
            st.markdown(
                '<div class="composer-hint"><strong>Enter</strong> to send &middot; <strong>Shift+Enter</strong> for a new line</div>',
                unsafe_allow_html=True,
            )

        pending_response_slot = None

    with qa_side:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Q/A settings</div>', unsafe_allow_html=True)
        if st.session_state.uploaded_document_info:
            st.caption("Document active: Q/A stays inside the uploaded file only.")
        else:
            st.caption("Corpus mode: strongest support in consumer, tax, service, education, motor-accident, and RTI-style matters.")
        if not st.session_state.uploaded_document_info:
            top_settings_left, top_settings_right = st.columns(2, gap="small")
            with top_settings_left:
                st.selectbox("Answer mode", QA_RETRIEVAL_PROFILES, key="chat_retrieval_profile")
            with top_settings_right:
                st.selectbox("Case type", CASE_TYPES, key="chat_case_type")
            st.text_area(
                "Context note",
                key="chat_context_note",
                height=68,
                placeholder="Optional short factual context for retrieval.",
            )
            with st.expander("Advanced filters", expanded=False):
                advanced_left, advanced_right = st.columns(2, gap="small")
                with advanced_left:
                    st.selectbox("Search mode", CHAT_SCOPES, key="chat_scope")
                with advanced_right:
                    st.selectbox("Forum / court type", FORUM_TYPES, key="chat_forum")
        if st.session_state.analysis_result:
            st.markdown("---")
            st.markdown("**Session controls**")
            if st.session_state.uploaded_document_info:
                st.caption("Review cases are ignored while a document is attached because Q/A is document-only.")
            else:
                if st.button("Use latest review cases", key="qa_use_review_cases", use_container_width=True):
                    queue_chat_context_from_triage(use_review_cases=True)
                    st.session_state.flash_notice = "Q/A will use the latest review cases."
                    st.rerun()
                if st.button("Copy review context", key="qa_copy_review_context", use_container_width=True):
                    queue_chat_context_from_triage()
                    st.session_state.flash_notice = "Q/A filters were updated from the latest review."
                    st.rerun()
        else:
            st.caption("Run a case review if you want to reuse its cases or filters here.")
        st.markdown("</div>", unsafe_allow_html=True)

    if chat_submission:
        question_text, _ = parse_chat_submission(chat_submission)
        if question_text.strip():
            with messages_container:
                render_user_message(question_text.strip())
                pending_response_slot = st.empty()
        elif pending_response_slot is None:
            pending_response_slot = messages_container.empty()
        process_chat_submission(api_url, chat_submission, 3, pending_response_slot)
        if st.session_state.qa_submit_error:
            with messages_container:
                st.error(st.session_state.qa_submit_error)
            st.session_state.qa_submit_error = None


with triage_tab:
    review_main, review_side = st.columns([3.55, 0.95], gap="large")
    with review_main:
        review_head_left, review_head_mid, review_head_right = st.columns([2.35, 0.82, 0.88])
        with review_head_left:
            st.markdown(
                """
                <div class="page-head">
                    <div class="page-title">Case Review</div>
                    <div class="page-copy">
                        Estimate likely case direction, compare outcome patterns, and move into Q/A when you need deeper authority analysis.
                    </div>
                    <div class="compact-disclaimer">First-pass triage only. Final strength depends on the full record and authority check.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with review_head_mid:
            run_analysis = st.button("Run review", key="run_case_review", type="primary", use_container_width=True)
        with review_head_right:
            if st.button("Clear results", key="review_clear_results", use_container_width=True):
                clear_results()
                st.rerun()

        st.markdown('<div class="surface">', unsafe_allow_html=True)
        triage_meta_1, triage_meta_2, triage_meta_3 = st.columns(3)
        with triage_meta_1:
            st.selectbox("Case type", CASE_TYPES, key="triage_case_type")
        with triage_meta_2:
            st.selectbox("Your role", USER_ROLES, key="triage_user_role")
        with triage_meta_3:
            st.selectbox("Forum / court type", FORUM_TYPES, key="triage_forum")
        st.text_area(
            "Core facts",
            key="triage_facts_input",
            height=96,
            placeholder="Summarize the dispute and why it reached court or a tribunal.",
        )
        triage_left, triage_right = st.columns(2)
        with triage_left:
            st.text_area(
                "Relief sought",
                key="triage_relief_input",
                height=72,
                placeholder="What order or remedy is being sought?",
            )
            st.text_area(
                "Opponent's main argument",
                key="triage_opponent_input",
                height=72,
                placeholder="What is the strongest point on the other side?",
            )
        with triage_right:
            st.text_area(
                "Evidence / documents",
                key="triage_evidence_input",
                height=72,
                placeholder="Key documents, records, or evidence that matter most.",
            )
            with st.expander("Additional narrative (optional)", expanded=False):
                st.text_area(
                    "Additional narrative",
                    key="triage_narrative_input",
                    height=72,
                    placeholder="Optional extra petition or order text.",
                    label_visibility="collapsed",
                )
        st.markdown("</div>", unsafe_allow_html=True)

        if run_analysis:
            if not has_substantive_case_input():
                st.error("Add a fuller facts summary or structured case details before running case review.")
            else:
                try:
                    with st.spinner("Running prediction, retrieval, and explanation..."):
                        st.session_state.analysis_result = call_prediction_api(
                            api_url,
                            build_triage_payload(
                                3,
                                st.session_state.ui_include_explanation,
                            ),
                        )
                        st.session_state.case_details = {}
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text
                    try:
                        detail = exc.response.json().get("detail", detail)
                    except Exception:
                        pass
                    st.error(f"Case review failed: {detail}")
                except Exception as exc:
                    st.error(f"Case review failed: {exc}")

        analysis_result = st.session_state.analysis_result
        if analysis_result:
            metric_1, metric_2, metric_3, metric_4 = st.columns(4)
            with metric_1:
                render_metric("Predicted direction", analysis_result["predicted_name"])
            with metric_2:
                render_metric("Likely effect for your side", analysis_result["favorability_label"])
            with metric_3:
                render_metric("Confidence", f'{analysis_result["confidence_score"]:.2f}%')
            with metric_4:
                render_metric("How much to rely on it", analysis_result["prediction_posture"])

            action_left, action_right = st.columns([1, 1])
            with action_left:
                if st.button("Use these cases in Q/A", key="review_to_chat_cases", use_container_width=True):
                    queue_chat_context_from_triage(use_review_cases=True)
                    st.session_state.flash_notice = "Q/A will use the latest review cases for the next answer."
                    st.rerun()
            with action_right:
                if st.button("Copy review filters into Q/A", key="review_to_chat_filters", use_container_width=True):
                    queue_chat_context_from_triage()
                    st.session_state.flash_notice = "Q/A filters were updated from the review inputs."
                    st.rerun()
            st.markdown(
                f'<div class="signal-line"><strong>{html.escape(build_review_signal(analysis_result))}</strong></div>',
                unsafe_allow_html=True,
            )

            triage_tab_1, triage_tab_2 = st.tabs(
                ["Summary", "Similar cases"]
            )
            with triage_tab_1:
                st.markdown('<div class="surface">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Review summary</div>', unsafe_allow_html=True)
                st.caption("Professional triage summary based on the current intake, retrieved authorities, and prediction signal.")
                st.markdown(analysis_result["explanation"] or "No explanation was requested.")
                with st.expander("Confidence breakdown", expanded=True):
                    render_probability_table(analysis_result["probabilities"])
                st.markdown("</div>", unsafe_allow_html=True)
            with triage_tab_2:
                render_similar_cases_only(analysis_result["workspace"], api_url, key_prefix="triage_workspace")

    with review_side:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Workspace</div>', unsafe_allow_html=True)
        st.markdown(make_status_pills(status_badges), unsafe_allow_html=True)
        if status_snapshot:
            st.markdown(
                (
                    f'<div class="rail-caption">{status_snapshot.get("retrieval_record_count", 0)} indexed cases'
                    f' &middot; {status_snapshot.get("qa_retrieval_record_count", 0)} indexed chunks</div>'
                ),
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Review settings</div>', unsafe_allow_html=True)
        st.caption("Case review uses a fixed shortlist of 3 authorities for consistency.")
        st.toggle("Generate review summary", key="ui_include_explanation")
        if st.session_state.uploaded_document_info:
            st.caption(
                f"Uploaded document available for review: {st.session_state.uploaded_document_info['filename']}"
            )
        st.markdown("</div>", unsafe_allow_html=True)

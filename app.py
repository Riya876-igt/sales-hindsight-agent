"""
DealCoach AI — Sales AI Agent
Streamlit UI + Groq (ultra-fast LLM inference) + Hindsight Cloud (persistent long-term memory)

Run with: streamlit run app.py
"""

import os
import time
import traceback
from datetime import datetime

import streamlit as st
from openai import OpenAI

try:
    from hindsight_client import Hindsight
    HINDSIGHT_SDK_AVAILABLE = True
except ImportError:
    HINDSIGHT_SDK_AVAILABLE = False


# =========================================================
# 1. CONFIGURATION
# =========================================================
# NOTE: hardcoding keys is fine for a hackathon demo, but rotate this key
# before pushing the repo anywhere public — os.getenv lets you override
# it with an environment variable without touching the code.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
HINDSIGHT_API_KEY = os.getenv("HINDSIGHT_API_KEY", "YOUR_HINDSIGHT_API_KEY")

GROQ_MODEL = "llama-3.3-70b-versatile"  # fallback if rate-limited: "llama-3.1-8b-instant"
HINDSIGHT_BASE_URL = "https://api.hindsight.vectorize.io"

st.set_page_config(
    page_title="DealCoach AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. STYLING — dark, enterprise SaaS, rounded, decluttered
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    [data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #232733;
    }
    div[data-testid="stMetric"] {
        background-color: #161b26;
        border: 1px solid #232733;
        border-radius: 14px;
        padding: 16px 18px;
    }
    .pipeline-badge {
        display: inline-block;
        width: 100%;
        text-align: center;
        background: linear-gradient(90deg, #1f2937, #111827);
        border: 1px solid #2d3444;
        border-radius: 10px;
        padding: 8px 10px;
        font-size: 12.5px;
        letter-spacing: 0.3px;
        color: #9ca3af;
        margin-top: 6px;
        margin-bottom: 4px;
    }
    .pipeline-badge b { color: #e6e6e6; }
    .response-card {
        background-color: #161b26;
        border: 1px solid #232733;
        border-radius: 14px;
        padding: 18px 20px;
        margin-top: 6px;
        line-height: 1.55;
    }
    .mem-badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.4px;
        padding: 3px 9px;
        border-radius: 999px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-pricing { background: #3b2412; color: #f5a35c; border: 1px solid #7c4a1e; }
    .badge-stakeholder { background: #12233b; color: #6cb2f5; border: 1px solid #1e4a7c; }
    .badge-competitor { background: #3b1212; color: #f56c6c; border: 1px solid #7c1e1e; }
    .badge-general { background: #1c1f2b; color: #9ca3af; border: 1px solid #2d3444; }
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #2d3444;
        background-color: #1a1f2b;
        color: #e6e6e6;
    }
    .stButton > button:hover {
        border-color: #4b5563;
        color: #ffffff;
    }
    div[data-testid="stChatInput"] { border-radius: 12px; }
    h1, h2, h3 { font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. SESSION STATE
# =========================================================
defaults = {
    "deal_name": "Acme Corp",
    "bank_id": "acme_corp_deal",
    "chat_history": [],          # list of {"role": ..., "content": ...}
    "interactions_logged": 3,    # starts at 3 to match the default metric card
    "last_memories": [],         # last retrieved memory blocks, for the inspector tab
    "pending_prompt": None,      # prompt queued by a quick-action pill or chat input
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================
# 4. CLIENT INITIALIZATION (cached, graceful failure)
# =========================================================
@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str):
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


@st.cache_resource(show_spinner=False)
def get_hindsight_client(api_key: str):
    if not HINDSIGHT_SDK_AVAILABLE:
        raise RuntimeError(
            "hindsight_client package is not installed. Run `pip install hindsight-client`."
        )
    return Hindsight(base_url=HINDSIGHT_BASE_URL, api_key=api_key)


groq_client, groq_error = None, None
try:
    groq_client = get_groq_client(GROQ_API_KEY)
except Exception as e:
    groq_error = str(e)

hindsight_client, hindsight_error = None, None
try:
    hindsight_client = get_hindsight_client(HINDSIGHT_API_KEY)
except Exception as e:
    hindsight_error = str(e)


# =========================================================
# 5. HELPERS
# =========================================================
def categorize_memory(text: str) -> str:
    """Very lightweight keyword tagging for the memory inspector demo."""
    t = text.lower()
    if any(k in t for k in ["price", "pricing", "budget", "discount", "cost"]):
        return "PRICING"
    if any(k in t for k in ["apex", "competitor", "rival", "alternative vendor"]):
        return "COMPETITOR"
    if any(k in t for k in ["cfo", "marcus", "stakeholder", "champion", "decision maker"]):
        return "STAKEHOLDER"
    return "GENERAL"


def badge_html(category: str) -> str:
    cls = {
        "PRICING": "badge-pricing",
        "COMPETITOR": "badge-competitor",
        "STAKEHOLDER": "badge-stakeholder",
        "GENERAL": "badge-general",
    }.get(category, "badge-general")
    return f'<span class="mem-badge {cls}">[{category}]</span>'


def extract_memory_texts(recall_result) -> list:
    """Safely pull plain-text memory blocks out of whatever shape Hindsight returns."""
    texts = []
    try:
        if recall_result is None:
            return texts
        # common shapes: object with .memories / .results, or a plain list/dict
        candidates = None
        for attr in ("memories", "results", "items"):
            if hasattr(recall_result, attr):
                candidates = getattr(recall_result, attr)
                break
        if candidates is None and isinstance(recall_result, dict):
            candidates = recall_result.get("memories") or recall_result.get("results")
        if candidates is None and isinstance(recall_result, list):
            candidates = recall_result

        if candidates:
            for item in candidates:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    texts.append(item.get("content") or item.get("text") or str(item))
                else:
                    texts.append(
                        getattr(item, "content", None)
                        or getattr(item, "text", None)
                        or str(item)
                    )
        elif isinstance(recall_result, str):
            texts.append(recall_result)
    except Exception:
        texts.append(str(recall_result))
    return [t for t in texts if t]


def safe_recall(bank_id: str, query: str) -> list:
    if hindsight_client is None:
        raise RuntimeError(hindsight_error or "Hindsight client not initialized.")
    result = hindsight_client.recall(bank_id=bank_id, query=query)
    return extract_memory_texts(result)


def safe_retain(bank_id: str, content: str, context: str):
    if hindsight_client is None:
        raise RuntimeError(hindsight_error or "Hindsight client not initialized.")
    return hindsight_client.retain(bank_id=bank_id, content=content, context=context)


def run_strategy_pipeline(prompt: str, bank_id: str, deal_name: str):
    """Recall -> Generate. Appends to chat_history and updates last_memories."""
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    memories = []
    with st.spinner("Querying Hindsight memory layer..."):
        try:
            memories = safe_recall(bank_id, prompt)
            st.session_state.last_memories = memories
        except Exception as e:
            st.session_state.last_memories = []
            st.error(f"Hindsight recall failed: {e}")

    memory_block = "\n".join(f"- {m}" for m in memories) if memories else "No prior memory found for this deal yet."

    system_prompt = (
        "You are DealCoach AI, an elite enterprise sales coach embedded in a CRM. "
        f"You are advising the rep on the deal with '{deal_name}'. "
        "Ground every recommendation strictly in the RETRIEVED MEMORY below — do not invent facts, "
        "names, numbers, or history that isn't present in it. If the memory is empty or thin, say so "
        "plainly and give general best-practice guidance instead. Be concise, concrete, and actionable: "
        "use short paragraphs or tight bullet points, and always end with a clear next step.\n\n"
        f"RETRIEVED MEMORY:\n{memory_block}"
    )

    answer = None
    with st.spinner("Generating strategy..."):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=700,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"⚠️ Groq generation failed: {e}"

    st.session_state.chat_history.append({"role": "assistant", "content": answer})


MOCK_HISTORY = [
    {
        "content": (
            "Discovery call: Acme Corp's procurement lead flagged that our proposed price is "
            "18% above their allocated Q3 budget. They asked for a phased rollout to spread cost "
            "across two fiscal quarters instead of a discount."
        ),
        "context": "Sales call discovery note",
    },
    {
        "content": (
            "Competitive note: Acme is also evaluating Apex Systems. Apex is quoting a lower "
            "upfront price but lacks native SSO and has weaker support SLAs. Rep should lead with "
            "total-cost-of-ownership and integration speed."
        ),
        "context": "Sales call discovery note",
    },
    {
        "content": (
            "Stakeholder note: CFO Marcus Webb is the economic buyer and is highly ROI-driven — he "
            "wants a payback-period model, not feature lists. He prefers concise written briefings "
            "over live demos and responds well to peer-company case studies."
        ),
        "context": "Sales call discovery note",
    },
]


# =========================================================
# 6. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ Deal Configuration")

    st.session_state.deal_name = st.text_input("Deal / Client Name", value=st.session_state.deal_name)
    st.session_state.bank_id = st.text_input("Hindsight Bank ID", value=st.session_state.bank_id)

    if st.button("🗑️ Clear Deal Memory", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_memories = []
        st.session_state.interactions_logged = 0
        st.toast("Local deal state cleared.", icon="🗑️")
        st.rerun()

    st.markdown(
        '<div class="pipeline-badge">Pipeline: <b>Retain</b> → <b>Recall</b> → <b>Generate</b></div>',
        unsafe_allow_html=True,
    )

    if groq_error:
        st.caption(f"⚠️ Groq: {groq_error}")
    if hindsight_error:
        st.caption(f"⚠️ Hindsight: {hindsight_error}")


# =========================================================
# 7. TOP METRICS BAR
# =========================================================
st.title("🧠 DealCoach AI")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Active Deal", st.session_state.deal_name)
with m2:
    st.metric("Memory Engine", "Hindsight Cloud (Connected)" if hindsight_client else "Hindsight Cloud (Offline)")
with m3:
    st.metric("Memory Status", f"{st.session_state.interactions_logged} Interactions Logged")

st.write("")


# =========================================================
# 8. TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(
    ["💬 Strategy Chat & Action Pills", "📝 Log Interaction & Notes", "🧠 Hindsight Memory Inspector"]
)

# ---------------------------------------------------------
# TAB 1 — Strategy Chat & Action Pills
# ---------------------------------------------------------
with tab1:
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("✉️ Draft Follow-up Email", use_container_width=True):
            st.session_state.pending_prompt = (
                f"Draft a concise, professional follow-up email to {st.session_state.deal_name} "
                "based on our history with this deal."
            )
    with p2:
        if st.button("💰 Handle Price & Competitor Objections", use_container_width=True):
            st.session_state.pending_prompt = (
                "Give me talking points to handle price pushback and competitor comparisons for this deal."
            )
    with p3:
        if st.button("📋 Generate Executive Briefing for CFO", use_container_width=True):
            st.session_state.pending_prompt = (
                "Generate a concise executive briefing for this deal's CFO, focused on ROI and payback period."
            )

    st.write("")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(f'<div class="response-card">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

    typed_prompt = st.chat_input("Ask DealCoach AI for strategy...")
    if typed_prompt:
        st.session_state.pending_prompt = typed_prompt

    if st.session_state.pending_prompt:
        prompt_to_run = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        if groq_client is None:
            st.error(f"Cannot generate — Groq client unavailable: {groq_error}")
        else:
            run_strategy_pipeline(prompt_to_run, st.session_state.bank_id, st.session_state.deal_name)
            st.rerun()

# ---------------------------------------------------------
# TAB 2 — Log Interaction & Notes
# ---------------------------------------------------------
with tab2:
    st.markdown("#### Log a new interaction")
    note_text = st.text_area(
        "Call notes, objections, budget caps, or stakeholder feedback",
        height=160,
        placeholder="e.g. Stakeholder mentioned a hard budget ceiling of $85k for this fiscal year...",
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("📌 Log Interaction", type="primary", use_container_width=True):
            if not note_text.strip():
                st.warning("Add some notes before logging.")
            else:
                try:
                    safe_retain(
                        st.session_state.bank_id,
                        note_text.strip(),
                        context="Sales call discovery note",
                    )
                    st.session_state.interactions_logged += 1
                    st.toast("✅ Interaction retained in Hindsight Cloud!", icon="🧠")
                except Exception as e:
                    st.error(f"Failed to retain interaction: {e}")

    with c2:
        if st.button("🎬 Simulate Past Deal History", use_container_width=True):
            success_count = 0
            with st.spinner("Injecting mock deal history into Hindsight..."):
                for entry in MOCK_HISTORY:
                    try:
                        safe_retain(st.session_state.bank_id, entry["content"], entry["context"])
                        success_count += 1
                    except Exception as e:
                        st.error(f"Failed to inject mock memory: {e}")
                        break
            if success_count:
                st.session_state.interactions_logged += success_count
                st.toast(f"✅ Injected {success_count} mock interactions!", icon="🧠")

    st.caption(f"Bank ID: `{st.session_state.bank_id}`  ·  Deal: `{st.session_state.deal_name}`")

# ---------------------------------------------------------
# TAB 3 — Hindsight Memory Inspector
# ---------------------------------------------------------
with tab3:
    st.markdown("#### Inspect raw memory retrieved for this deal")

    inspect_query = st.text_input(
        "Search query",
        value="deal history summary",
        label_visibility="collapsed",
        placeholder="Search Hindsight memory...",
    )
    if st.button("🔍 Fetch Memories", use_container_width=True):
        try:
            with st.spinner("Querying Hindsight memory layer..."):
                st.session_state.last_memories = safe_recall(st.session_state.bank_id, inspect_query)
        except Exception as e:
            st.error(f"Hindsight recall failed: {e}")

    memories = st.session_state.last_memories
    if not memories:
        st.info("No memories loaded yet — run a chat query or fetch memories above.")
    else:
        for i, mem in enumerate(memories):
            category = categorize_memory(mem)
            with st.expander(f"Memory block #{i + 1}", expanded=False):
                st.markdown(badge_html(category), unsafe_allow_html=True)
                st.write(mem)

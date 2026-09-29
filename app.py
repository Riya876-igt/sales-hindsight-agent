"""
OutStride AI — memory-first sales intelligence
Streamlit (UI)  +  Groq (LLM + Whisper voice-to-text)  +  Hindsight Cloud (long-term memory)

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import hmac
import html
import inspect
import io
import json
import os
import re
import secrets
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import streamlit as st
from openai import OpenAI

try:
    from hindsight_client import Hindsight
    HINDSIGHT_SDK = True
except ImportError:  # the app still loads; Hindsight features report a clear error
    Hindsight = None
    HINDSIGHT_SDK = False


# =====================================================================
# 1. CONFIGURATION
# =====================================================================
# Hackathon convenience: keys are defaults here, override with env vars any time.
# ROTATE BOTH KEYS before this repo goes public.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HINDSIGHT_API_KEY = os.getenv("HINDSIGHT_API_KEY", "")
HINDSIGHT_BASE_URL = "https://api.hindsight.vectorize.io"

# llama-3.3-70b / llama-3.1-8b were deprecated by Groq on 2026-08-16.
_primary = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_MODELS = [_primary] + [m for m in ("openai/gpt-oss-20b",) if m != _primary]
STT_MODEL = "whisper-large-v3-turbo"  # voice -> text

REGISTRY_BANK = "outstride_users"  # write-only directory of accounts, visible in the Hindsight console
DATA_DIR = Path(__file__).resolve().parent / "outstride_data"
USERS_FILE = DATA_DIR / "users.json"

PRODUCT_CHUNK_CHARS = 10_000   # characters per retain() call for product documents
PRODUCT_MAX_CHUNKS = 30        # per document (~300k characters)
FILE_TYPES = ["pdf", "docx", "txt", "md", "csv"]
STAGES = ["Discovery", "Demo", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
INTERESTS = ["High", "Medium", "Low"]

ROLES = [
    ("Sales Representative", "Prospect, pitch and close your own deals"),
    ("Sales Manager", "Coach a team and review the pipeline"),
    ("Account Executive", "Own key accounts and negotiations"),
    ("Business Development", "Open new markets and partnerships"),
    ("Founder / Entrepreneur", "Sell your own product or service"),
    ("Customer Success", "Retain and grow existing customers"),
    ("HR / Recruiter", "Manage candidates, clients and stakeholders"),
    ("Business Employee", "Track daily work with clients and partners"),
    ("Other", "Something else"),
]

st.set_page_config(page_title="OutStride AI", page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")


# ---- Streamlit version compatibility (width kwarg + chat_input capabilities) ----
def _stretch(fn):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


STRETCH_BTN = _stretch(st.button)
STRETCH_POP = _stretch(st.popover)
STRETCH_FORM = _stretch(st.form_submit_button)
try:
    _CI_PARAMS = set(inspect.signature(st.chat_input).parameters)
except (TypeError, ValueError):
    _CI_PARAMS = set()
CI_FILES = "accept_file" in _CI_PARAMS
CI_AUDIO = "accept_audio" in _CI_PARAMS


# =====================================================================
# 2. SMALL UTILITIES
# =====================================================================
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def short_id() -> str:
    return uuid.uuid4().hex[:8]


def initials(name: str) -> str:
    parts = [p for p in re.split(r"\s+", (name or "").strip()) if p]
    return ("".join(p[0] for p in parts[:2]) or "?").upper()


def fmt_day(d: str) -> str:
    try:
        dd = date.fromisoformat(d)
    except Exception:
        return d or "—"
    label = f"{dd:%b} {dd.day}"
    return label if dd.year == date.today().year else f"{label}, {dd.year}"


def rel_day(d: str | None) -> str:
    if not d:
        return "—"
    try:
        delta = (date.today() - date.fromisoformat(d)).days
    except Exception:
        return "—"
    if delta <= 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if delta < 7:
        return f"{delta} days ago"
    return fmt_day(d)


def clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def clean_list(value, limit: int = 6) -> list[str]:
    if isinstance(value, str):
        value = [v for v in re.split(r"[,;\n]", value)]
    out, seen = [], set()
    for v in value or []:
        s = str(v).strip().strip("-•* ").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(clip(s, 60))
    return out[:limit]


def parse_json(text: str | None):
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _g(obj, *names):
    """Read the first available field from a dict or an object."""
    for n in names:
        if isinstance(obj, dict) and obj.get(n) not in (None, ""):
            return obj[n]
        v = getattr(obj, n, None)
        if v not in (None, ""):
            return v
    return None


# =====================================================================
# 3. LOCAL ACCOUNT + WORKSPACE STORE
#    (accounts/workspaces live in ./outstride_data; everything that matters
#     to the AI is ALSO retained in Hindsight)
# =====================================================================
def _read_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _write_json(path: Path, data) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_users() -> dict:
    return _read_json(USERS_FILE, {})


def save_users(users: dict) -> None:
    _write_json(USERS_FILE, users)


def user_key(email: str) -> str:
    return hashlib.sha1(email.strip().lower().encode()).hexdigest()[:10]


def bank_for(email: str) -> str:
    return f"outstride_{user_key(email)}"


def hash_pw(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 200_000).hex()


def new_ws() -> dict:
    return {"customers": {}, "selected_customer": None, "product": None, "chats": []}


def load_ws(email: str) -> dict:
    ws = _read_json(DATA_DIR / f"ws_{user_key(email)}.json", None)
    if not isinstance(ws, dict):
        ws = new_ws()
    for k, v in new_ws().items():
        ws.setdefault(k, v)
    return ws


def save_ws() -> None:
    email = st.session_state.get("auth_email")
    if email and st.session_state.get("ws") is not None:
        _write_json(DATA_DIR / f"ws_{user_key(email)}.json", st.session_state.ws)


def current_user() -> dict | None:
    email = st.session_state.get("auth_email")
    return load_users().get(email) if email else None


def create_account(name: str, email: str, password: str, company: str):
    email = email.strip().lower()
    if not name.strip():
        return False, "Please enter your name."
    if not EMAIL_RE.match(email):
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    users = load_users()
    if email in users:
        return False, "An account with this email already exists — try logging in."
    salt = secrets.token_hex(16)
    users[email] = {
        "name": name.strip(), "email": email, "company": company.strip(),
        "salt": salt, "pw": hash_pw(password, salt),
        "profession": "", "created": now_iso(), "bank_id": bank_for(email),
    }
    save_users(users)
    return True, email


def check_login(email: str, password: str):
    email = email.strip().lower()
    u = load_users().get(email)
    if not u or not hmac.compare_digest(u["pw"], hash_pw(password, u["salt"])):
        return False, "Incorrect email or password."
    return True, email


def update_user(email: str, **fields) -> None:
    users = load_users()
    if email in users:
        users[email].update(fields)
        save_users(users)


def init_state() -> None:
    defaults = {
        "auth_email": None, "page": "home", "ws": None, "chat_id": short_id(),
        "chat_messages": [], "pending_update": None, "last_saved": None,
        "recall_state": None, "role_choice": None, "editing_role": False,
        "hs_ok": None, "banks_ready": set(),
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def go(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def sign_in(email: str) -> None:
    st.session_state.auth_email = email
    st.session_state.ws = load_ws(email)
    st.session_state.chat_id = short_id()
    st.session_state.chat_messages = []
    st.session_state.pending_update = None
    st.session_state.last_saved = None
    st.session_state.recall_state = None
    st.session_state.page = "home"


def sign_out() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]


# =====================================================================
# 4. CLIENTS: Groq (LLM + Whisper) and Hindsight (memory)
# =====================================================================
@st.cache_resource(show_spinner=False)
def get_groq(api_key: str):
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


@st.cache_resource(show_spinner=False)
def get_hindsight(api_key: str):
    if not HINDSIGHT_SDK:
        raise RuntimeError("hindsight-client is not installed — run: pip install hindsight-client")
    return Hindsight(base_url=HINDSIGHT_BASE_URL, api_key=api_key)


groq_client, groq_err = None, None
try:
    groq_client = get_groq(GROQ_API_KEY)
except Exception as e:  # pragma: no cover
    groq_err = str(e)

hindsight, hindsight_err = None, None
try:
    hindsight = get_hindsight(HINDSIGHT_API_KEY)
except Exception as e:
    hindsight_err = str(e)


def hs_state_label() -> tuple[str, str]:
    """(label, css-class) describing Hindsight connectivity for the UI."""
    if hindsight is None:
        return "Not configured", "red"
    ok = st.session_state.get("hs_ok")
    if ok is False:
        return "Sync error", "amber"
    return "Connected", "green"


def ensure_bank(bank_id: str, name: str) -> None:
    ready = st.session_state.setdefault("banks_ready", set())
    if bank_id in ready or hindsight is None:
        return
    try:
        hindsight.create_bank(bank_id=bank_id, name=name)
    except Exception:
        pass  # already exists (or auto-created); real problems surface on retain/recall
    ready.add(bank_id)


def hs_retain(bank_id: str, content: str, context: str, ts: str | None = None, bank_name: str = "OutStride memory"):
    if hindsight is None:
        raise RuntimeError(hindsight_err or "Hindsight is not available")
    ensure_bank(bank_id, bank_name)
    kwargs = dict(bank_id=bank_id, content=content, context=context)
    if ts:
        kwargs["timestamp"] = ts
    try:
        try:
            hindsight.retain(**kwargs)
        except TypeError:
            kwargs.pop("timestamp", None)
            hindsight.retain(**kwargs)
        st.session_state.hs_ok = True
    except Exception:
        st.session_state.hs_ok = False
        raise


def hs_recall(bank_id: str, query: str, limit: int = 12) -> list[dict]:
    if hindsight is None:
        raise RuntimeError(hindsight_err or "Hindsight is not available")
    ensure_bank(bank_id, "OutStride memory")
    try:
        res = hindsight.recall(bank_id=bank_id, query=query)
        st.session_state.hs_ok = True
    except Exception:
        st.session_state.hs_ok = False
        raise
    items = _g(res, "results", "memories", "items")
    if items is None and isinstance(res, list):
        items = res
    out = []
    for it in items or []:
        text = it if isinstance(it, str) else _g(it, "text", "content", "memory")
        if not text:
            continue
        out.append({
            "text": str(text),
            "type": "" if isinstance(it, str) else str(_g(it, "type", "fact_type") or ""),
            "when": "" if isinstance(it, str) else str(_g(it, "mentioned_at", "occurred_start", "date") or ""),
        })
    return out[:limit]


def llm(messages: list[dict], max_tokens: int = 1600, temperature: float = 0.3) -> str:
    if groq_client is None:
        raise RuntimeError(groq_err or "Groq client is not available")
    last = None
    for model in LLM_MODELS:
        kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if model.startswith("openai/gpt-oss"):
            kwargs["extra_body"] = {"reasoning_effort": "low"}  # keep reasoning tokens small
        try:
            resp = groq_client.chat.completions.create(**kwargs)
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content
            last = RuntimeError("the model returned an empty response")
        except Exception as e:
            last = e
    raise last or RuntimeError("LLM call failed")


def transcribe(audio) -> str:
    if groq_client is None:
        raise RuntimeError(groq_err or "Groq client is not available")
    data = audio.getvalue() if hasattr(audio, "getvalue") else audio.read()
    if not data:
        raise RuntimeError("No audio was captured — check microphone permission in your browser.")
    name = getattr(audio, "name", None) or "voice.wav"
    mime = getattr(audio, "type", None) or "audio/wav"
    resp = groq_client.audio.transcriptions.create(
        model=STT_MODEL, file=(name, data, mime), response_format="text")
    return (resp if isinstance(resp, str) else getattr(resp, "text", "") or str(resp)).strip()


# =====================================================================
# 5. DOCUMENT HANDLING (product documents & chat attachments)
# =====================================================================
def extract_text(uploaded) -> str:
    name = (uploaded.name or "").lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("PDF support needs: pip install pypdf")
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages).strip()
    if name.endswith(".docx"):
        try:
            import docx
        except ImportError:
            raise RuntimeError("DOCX support needs: pip install python-docx")
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs if p.text.strip()]
        for table in d.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        return "\n".join(parts).strip()
    return data.decode("utf-8", errors="ignore").strip()


def chunk_text(text: str, size: int = PRODUCT_CHUNK_CHARS) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size:  # very long paragraph: hard split
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:size])
            p = p[size:]
        if len(cur) + len(p) + 2 > size and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


# =====================================================================
# 6. DEAL MEMORY: customers, interactions, snapshots
# =====================================================================
def ws() -> dict:
    return st.session_state.ws


def selected_customer() -> str | None:
    return ws().get("selected_customer")


def resolve_customer(name: str) -> str:
    name = (name or "").strip()
    for existing in ws()["customers"]:
        if existing.lower() == name.lower():
            return existing
    return name


def create_customer(name: str) -> str | None:
    name = resolve_customer(name)
    if not name:
        return None
    ws()["customers"].setdefault(name, {"created": now_iso(), "interactions": []})
    ws()["selected_customer"] = name
    save_ws()
    return name


def customer_snapshot(name: str | None) -> dict:
    c = ws()["customers"].get(name) if name else None
    inter = sorted(c["interactions"], key=lambda r: (r.get("date", ""), r.get("ts", ""))) if c else []

    def collect(field):
        seen, out = set(), []
        for r in inter:
            for v in r.get(field, []) or []:
                k = v.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    out.append(v.strip())
        return out

    def latest(field):
        return next((r[field] for r in reversed(inter) if r.get(field)), "")

    return {
        "interactions": inter, "count": len(inter),
        "needs": collect("needs"), "objections": collect("objections"), "competitors": collect("competitors"),
        "stage": latest("stage"), "interest": latest("interest"), "next_action": latest("next_action"),
        "last": inter[-1]["date"] if inter else None,
        "unsynced": sum(1 for r in inter if not r.get("synced")),
    }


def timeline_text(name: str) -> str:
    lines = []
    for r in customer_snapshot(name)["interactions"]:
        bits = [f"{r['date']} — {r['title']}: {r['summary']}"]
        for label, key in (("needs", "needs"), ("objections", "objections"), ("competitors", "competitors")):
            if r.get(key):
                bits.append(f"{label}: {', '.join(r[key])}")
        if r.get("next_action"):
            bits.append(f"next: {r['next_action']}")
        lines.append(" | ".join(bits))
    return "\n".join(lines)


def memory_text(customer: str, rec: dict) -> str:
    lines = [f"Customer: {customer}", f"Date: {rec['date']}", f"Event: {rec['title']}"]
    body = clip(rec.get("raw") or rec.get("summary") or "", 3000)
    lines.append(f"Update: {body}")
    if rec.get("summary") and rec.get("raw") and rec["summary"] != rec["raw"]:
        lines.append(f"Summary: {rec['summary']}")
    for label, key in (("Customer needs", "needs"), ("Objections", "objections"),
                       ("Competitors mentioned", "competitors")):
        if rec.get(key):
            lines.append(f"{label}: {', '.join(rec[key])}")
    if rec.get("next_action"):
        lines.append(f"Next action: {rec['next_action']}")
    if rec.get("stage"):
        lines.append(f"Deal stage: {rec['stage']}")
    if rec.get("interest"):
        lines.append(f"Customer interest: {rec['interest']}")
    return "\n".join(lines)


def record_ts(rec: dict) -> str:
    return now_iso() if rec["date"] == date.today().isoformat() else f"{rec['date']}T10:00:00Z"


def sync_interaction(customer: str, rec: dict):
    try:
        hs_retain(current_user()["bank_id"], memory_text(customer, rec),
                  context=f"Sales update for customer {customer}", ts=record_ts(rec))
        return True, None
    except Exception as e:
        return False, str(e)


def commit_update(customer: str, rec: dict) -> bool:
    customer = resolve_customer(customer)
    if not customer:
        return False
    c = ws()["customers"].setdefault(customer, {"created": now_iso(), "interactions": []})
    rec["synced"] = False
    c["interactions"].append(rec)
    ws()["selected_customer"] = customer
    with st.spinner("Saving to Hindsight memory…"):
        ok, err = sync_interaction(customer, rec)
    rec["synced"] = ok
    save_ws()
    st.session_state.last_saved = {"customer": customer, "rec": rec, "ok": ok, "err": err}
    return True


def retry_sync(customer: str) -> None:
    fails = 0
    with st.spinner("Re-syncing to Hindsight…"):
        for rec in ws()["customers"].get(customer, {}).get("interactions", []):
            if not rec.get("synced"):
                ok, _ = sync_interaction(customer, rec)
                rec["synced"] = ok
                fails += 0 if ok else 1
    save_ws()
    (st.toast("Everything is synced to Hindsight.", icon="🧠") if not fails
     else st.toast(f"{fails} item(s) still failed to sync.", icon="⚠️"))


# =====================================================================
# 7. UNDERSTANDING A MESSAGE (update vs question) + extraction
# =====================================================================
QUESTION_STARTS = ("what", "who", "when", "where", "why", "how", "which", "show", "summar", "prepare",
                   "create", "draft", "give", "list", "tell me", "can you", "could you", "explain",
                   "remind", "help me", "write")


def looks_like_question(t: str) -> bool:
    t = t.strip().lower()
    return t.endswith("?") or t.startswith(QUESTION_STARTS)


def normalize_analysis(obj, text: str, force_update: bool) -> dict:
    obj = obj if isinstance(obj, dict) else {}
    intent = str(obj.get("intent", "")).lower()
    if intent not in ("update", "question"):
        intent = "question" if looks_like_question(text) else "update"
    if force_update:
        intent = "update"
    stage = next((s for s in STAGES if s.lower() == str(obj.get("deal_stage", "")).strip().lower()), "")
    interest = next((s for s in INTERESTS if s.lower() == str(obj.get("interest", "")).strip().lower()), "")
    try:
        when = date.fromisoformat(str(obj.get("date", "")))
        if when > date.today():
            when = date.today()
    except Exception:
        when = date.today()
    cust = obj.get("customer")
    return {
        "intent": intent,
        "customer": str(cust).strip() if cust and str(cust).lower() != "null" else "",
        "title": clip(str(obj.get("title") or "Daily update"), 60),
        "summary": clip(str(obj.get("summary") or text), 300),
        "date": when.isoformat(),
        "needs": clean_list(obj.get("needs")),
        "objections": clean_list(obj.get("objections")),
        "competitors": clean_list(obj.get("competitors")),
        "next_action": clip(str(obj.get("next_action") or ""), 160),
        "stage": stage, "interest": interest,
    }


def analyze_message(text: str, profession: str, force_update: bool = False) -> dict:
    known = ", ".join(ws()["customers"].keys()) or "none"
    system = f"""You are the understanding engine of OutStride AI, a memory-first assistant for a {profession}.
Today is {date.today().isoformat()}. Known customers: {known}. Currently selected customer: {selected_customer() or 'none'}.

Decide what the user's message is, then reply with ONLY a JSON object (no prose, no markdown fences):
{{
 "intent": "update" or "question",
 "customer": "the customer/company the message is about, or null",
 "title": "3-6 word timeline title such as 'Pricing discussion' (updates only)",
 "summary": "one clear sentence describing what happened (updates only)",
 "date": "YYYY-MM-DD when it happened; today unless the user says otherwise",
 "needs": ["short labels of what the customer needs"],
 "objections": ["short labels such as 'Pricing'"],
 "competitors": ["competitor names"],
 "next_action": "the next step, or empty string",
 "deal_stage": "Discovery, Demo, Proposal, Negotiation, Closed Won, Closed Lost or empty string",
 "interest": "High, Medium, Low or empty string"
}}
Rules: intent is "update" when the user reports something that happened, was said or was decided (calls,
meetings, emails, feedback, budgets, objections). It is "question" when the user asks for information,
analysis, a summary, a draft or advice. Only extract what the message actually states — never invent facts.
Use empty lists / empty strings when something is unknown."""
    if force_update:
        system += '\nThe user is logging an update, so "intent" MUST be "update".'
    try:
        raw = llm([{"role": "system", "content": system}, {"role": "user", "content": clip(text, 8000)}],
                  max_tokens=1200, temperature=0.0)
        obj = parse_json(raw)
    except Exception:
        obj = None
    return normalize_analysis(obj, text, force_update)


# =====================================================================
# 8. RECALL + GROUNDED ANSWERS + CALL BRIEFING
# =====================================================================
def categorize_memory(text: str, scope: str = "") -> str:
    t = text.lower()
    if scope == "product" or t.startswith("product") or "document:" in t[:120]:
        return "PRODUCT"
    if t.startswith("user profile"):
        return "PROFILE"
    if any(k in t for k in ("price", "pricing", "budget", "discount", "cost")):
        return "PRICING"
    if any(k in t for k in ("competitor", "compar", "rival", "alternative")):
        return "COMPETITOR"
    if any(k in t for k in ("cfo", "ceo", "stakeholder", "champion", "decision maker", "procurement")):
        return "STAKEHOLDER"
    return "GENERAL"


def gather_memory(customer: str | None, query: str, include_product: bool = True):
    """Recall from Hindsight. Returns (items, warnings)."""
    bank = current_user()["bank_id"]
    items, warns, seen = [], [], set()

    def add(found, scope):
        for it in found:
            key = it["text"].strip().lower()[:200]
            if key not in seen:
                seen.add(key)
                items.append({**it, "scope": scope})

    try:
        add(hs_recall(bank, f"{customer}: {query}" if customer else query), "customer")
    except Exception as e:
        warns.append(f"Hindsight recall failed: {e}")
    if include_product and customer and ws().get("product"):
        try:
            add(hs_recall(bank, f"Product information relevant to: {query}", limit=6), "product")
        except Exception as e:
            warns.append(f"Product recall failed: {e}")
    return items, warns


def system_prompt(customer: str | None, mem: list[dict]) -> str:
    user = current_user()
    snap = customer_snapshot(customer) if customer else None
    parts = [
        f"You are OutStride AI, a memory-first sales copilot for {user['name']}, a {user['profession'] or 'sales professional'}.",
        f"Today is {date.today().isoformat()}. Current customer: {customer or 'none selected'}.",
        "Ground every statement in the MEMORY below (Hindsight long-term memory, the deal timeline and product knowledge). "
        "Never invent names, numbers, dates or commitments that are not in memory. If something is missing, say so plainly "
        "and suggest what the user should capture next. Be concise and practical: short paragraphs or tight bullets, "
        "and finish with a clear next step when it helps.",
    ]
    if snap and snap["count"]:
        parts.append(
            "DEAL SNAPSHOT — stage: %s | interest: %s | needs: %s | objections: %s | competitors: %s | next action: %s\n"
            "DEAL TIMELINE:\n%s" % (
                snap["stage"] or "unknown", snap["interest"] or "unknown",
                ", ".join(snap["needs"]) or "none", ", ".join(snap["objections"]) or "none",
                ", ".join(snap["competitors"]) or "none", snap["next_action"] or "none",
                timeline_text(customer)))
    product = ws().get("product")
    if product:
        parts.append(f"PRODUCT: {product['name']} — {product.get('summary', '')}")
    if mem:
        parts.append("RECALLED MEMORY (Hindsight):\n" + "\n".join(f"- [{m['scope']}] {clip(m['text'], 700)}" for m in mem))
    else:
        parts.append("RECALLED MEMORY (Hindsight): nothing relevant was returned.")
    return "\n\n".join(parts)


def answer_question(customer: str | None, question: str, history: list[dict] | None = None):
    mem, warns = gather_memory(customer, question)
    msgs = [{"role": "system", "content": system_prompt(customer, mem)}]
    for m in (history or [])[-6:]:
        msgs.append({"role": m["role"], "content": clip(m["content"], 1200)})
    msgs.append({"role": "user", "content": question})
    return llm(msgs), mem, warns


def generate_briefing(customer: str):
    mem, warns = gather_memory(customer, "history, needs, objections, competitors, pricing, stakeholders and next steps")
    ask = (
        "Prepare a pre-call briefing for my next call with this customer using only what is in memory. "
        'Reply with ONLY a JSON object: {"headline": "one short line", "highlights": ["max 5 short bullets"], '
        '"discussion": ["3-5 suggested discussion points, in order"], "watch_outs": ["0-3 risks to watch"]}'
    )
    raw = llm([{"role": "system", "content": system_prompt(customer, mem)}, {"role": "user", "content": ask}],
              max_tokens=1600)
    data = parse_json(raw)
    if not isinstance(data, dict) or not (data.get("highlights") or data.get("discussion")):
        data = {"raw": raw}
    return data, mem, warns


# =====================================================================
# 9. PRODUCT KNOWLEDGE + DEMO DATA
# =====================================================================
def ingest_product(name: str, files: list) -> None:
    user = current_user()
    bank = user["bank_id"]
    product = ws().get("product") or {"name": name, "summary": "", "files": []}
    product["name"] = name
    total_ok, total_fail, sample, last_err = 0, 0, "", ""
    for f in files:
        try:
            text = extract_text(f)
        except Exception as e:
            st.error(f"Couldn't read {f.name}: {e}")
            continue
        if not text:
            st.warning(f"{f.name} has no readable text (scanned PDF?).")
            continue
        sample = sample or text[:12_000]
        chunks = chunk_text(text)
        truncated = len(chunks) > PRODUCT_MAX_CHUNKS
        chunks = chunks[:PRODUCT_MAX_CHUNKS]
        bar = st.progress(0.0, text=f"Storing {f.name} in Hindsight…")
        ok = 0
        for i, ch in enumerate(chunks, 1):
            try:
                hs_retain(bank, f"Product: {name}\nDocument: {f.name} (part {i} of {len(chunks)})\n\n{ch}",
                          context="Product documentation", ts=now_iso())
                ok += 1
            except Exception as e:
                total_fail += 1
                last_err = str(e)
            bar.progress(i / len(chunks), text=f"Storing {f.name} in Hindsight… ({i}/{len(chunks)})")
        bar.empty()
        total_ok += ok
        product["files"].append({"name": f.name, "chars": len(text), "chunks": len(chunks),
                                 "stored": ok, "ts": now_iso()})
        if truncated:
            st.warning(f"{f.name} is very large — the first {PRODUCT_MAX_CHUNKS * PRODUCT_CHUNK_CHARS:,} characters were stored.")
        if ok < len(chunks):
            st.error(f"{len(chunks) - ok} part(s) of {f.name} failed to store: {last_err}")
    if sample:
        try:
            with st.spinner("Summarising the product…"):
                product["summary"] = clip(llm([
                    {"role": "system", "content": "Summarise what this product/service is and who it is for in ONE sentence of at most 30 words. Use only the text provided."},
                    {"role": "user", "content": sample}], max_tokens=400, temperature=0.2), 260)
            hs_retain(bank, f"Product overview — {name}: {product['summary']}", context="Product documentation", ts=now_iso())
        except Exception:
            pass
    ws()["product"] = product
    save_ws()
    if total_ok:
        st.toast(f"Product knowledge stored in Hindsight ({total_ok} part(s)).", icon="🧠")


DEMO_CUSTOMER = "ABC Technologies"
DEMO_PRODUCT = {
    "name": "AI Automation Platform",
    "summary": "An AI workflow-automation platform that connects to CRM/ERP tools and removes manual reporting.",
    "memory": (
        "Product: AI Automation Platform (demo product). Automates manual reporting and back-office workflows with AI agents. "
        "Key features: no-code workflow builder, 40+ native integrations (CRM, ERP, email, Slack), SSO and role-based access, "
        "audit logs, 24/7 support with a 4-hour SLA. Typical implementation takes 4-6 weeks with a dedicated onboarding "
        "engineer. Pricing: Starter, Growth and Enterprise tiers; annual contracts, with phased rollout billing available. "
        "Differentiators: faster integration time and stronger support SLAs than lower-priced alternatives."
    ),
}
DEMO_TIMELINE = [
    (25, "Initial discovery call", "Customer is interested in automating manual reporting.",
     ["Automation", "Easy integration"], [], [], "Schedule a product demo", "Discovery", "Medium"),
    (18, "Product demo", "Customer liked the integration features.",
     ["Easy integration"], [], [], "Share pricing options", "Demo", "High"),
    (11, "Pricing discussion", "Pricing became an objection; budget is tight this quarter.",
     [], ["Pricing"], [], "Prepare a phased-rollout option", "Proposal", "Medium"),
    (6, "Competitor mentioned", "Customer is comparing XYZ, which quotes a lower upfront price.",
     [], ["Pricing", "Implementation time"], ["XYZ"], "Prepare a competitive comparison", "Negotiation", "Medium"),
    (1, "Latest update", "Customer asked for a revised proposal.",
     [], [], [], "Send revised proposal", "Negotiation", "High"),
]


def load_demo_deal() -> None:
    bank = current_user()["bank_id"]
    c = ws()["customers"].setdefault(DEMO_CUSTOMER, {"created": now_iso(), "interactions": []})
    existing = {r["title"] for r in c["interactions"]}
    bar = st.progress(0.0, text="Loading demo deal into Hindsight…")
    steps = len(DEMO_TIMELINE) + 1
    for i, (ago, title, summary, needs, objs, comps, nxt, stage, interest) in enumerate(DEMO_TIMELINE, 1):
        if title in existing:
            continue
        rec = {"id": short_id(), "date": (date.today() - timedelta(days=ago)).isoformat(), "ts": now_iso(),
               "title": title, "summary": summary, "raw": summary, "needs": needs, "objections": objs,
               "competitors": comps, "next_action": nxt, "stage": stage, "interest": interest,
               "source": "demo", "synced": False}
        c["interactions"].append(rec)
        rec["synced"], _ = sync_interaction(DEMO_CUSTOMER, rec)
        bar.progress(i / steps, text=f"Loading demo deal into Hindsight… ({i}/{steps})")
    if not ws().get("product"):
        try:
            hs_retain(bank, DEMO_PRODUCT["memory"], context="Product documentation", ts=now_iso())
        except Exception:
            pass
        ws()["product"] = {"name": DEMO_PRODUCT["name"], "summary": DEMO_PRODUCT["summary"], "files": []}
    bar.empty()
    ws()["selected_customer"] = DEMO_CUSTOMER
    save_ws()
    st.toast("Demo deal loaded.", icon="🧠")


# =====================================================================
# 10. CHAT SESSIONS (Chat Space)
# =====================================================================
def new_chat() -> None:
    st.session_state.chat_id = short_id()
    st.session_state.chat_messages = []
    st.session_state.pending_update = None
    st.session_state.last_saved = None


def save_chat() -> None:
    msgs = st.session_state.chat_messages
    if not msgs:
        return
    title = next((m["content"] for m in msgs if m["role"] == "user"), "New chat")
    payload = {"id": st.session_state.chat_id, "title": clip(title, 60),
               "customer": selected_customer(), "updated": now_iso(), "messages": msgs}
    chats = ws()["chats"]
    entry = next((c for c in chats if c["id"] == payload["id"]), None)
    if entry:
        entry.update(payload)
    else:
        payload["created"] = now_iso()
        chats.append(payload)
    ws()["chats"] = chats[-50:]
    save_ws()


# =====================================================================
# 11. STYLING — black + orange, minimal, premium B2B SaaS
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700;800&display=swap');
:root{
  --bg:#050505; --surface:#0f0f0f; --surface2:#181818; --border:#2b2119; --text:#f6efe8; --muted:#a59888;
  --accent:#ff7a1a; --accent2:#ff9f45; --accent3:#ffc078; --accent-soft:rgba(255,122,26,.14);
  --green:#ffb25e; --amber:#ff7a1a; --red:#ff4d2e;   /* status dots kept in the orange family */
  --radius:16px; --shadow:0 1px 2px rgba(0,0,0,.5),0 12px 32px rgba(0,0,0,.35);
}
html, body, .stApp, button, input, textarea { font-family:'Inter',system-ui,-apple-system,sans-serif; }
.stApp { background:var(--bg); color:var(--text); }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"],
[data-testid="stDecoration"], #MainMenu, footer { display:none !important; }
.block-container { max-width:1120px; padding:1rem 2rem 8rem; }
h1,h2,h3,h4 { letter-spacing:-.02em; }

/* ---------- brand (big, centred, animated) ---------- */
@keyframes shine { 0%{background-position:0% 50%} 100%{background-position:300% 50%} }
.brand, .auth-brand { font-family:'Sora','Inter',sans-serif; font-weight:800; text-align:center; letter-spacing:-.045em;
  background:linear-gradient(90deg,#ff5a00,#ff9f45,#ffd2a1,#ff7a1a,#ff5a00); background-size:300% 100%;
  -webkit-background-clip:text; background-clip:text; color:transparent; -webkit-text-fill-color:transparent;
  animation:shine 7s linear infinite; cursor:pointer; user-select:none;
  transition:transform .28s cubic-bezier(.2,.9,.3,1.4), filter .28s; }
.brand { font-size:2.7rem; line-height:1.1; }
.brand:hover, .auth-brand:hover { transform:scale(1.07) rotate(-.6deg); filter:drop-shadow(0 0 20px rgba(255,122,26,.65)); animation-duration:2s; }
.brand:active { transform:scale(.97); }

/* ---------- top bar ---------- */
.st-key-topbar { border-bottom:1px solid var(--border); padding:.1rem 0 .9rem; margin-bottom:1.2rem; }
.st-key-topbar [data-testid="stHorizontalBlock"] { align-items:center; flex-wrap:nowrap !important; }
.st-key-topbar [data-testid="stColumn"], .st-key-topbar [data-testid="column"] { min-width:0 !important; }
.st-key-topbar_right { display:flex; justify-content:flex-end; }
.st-key-topbar_right button { border-radius:999px; background:var(--surface); border:1px solid var(--border); }

/* ---------- sidebar (permanent) ---------- */
section[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--border);
  width:208px !important; min-width:208px !important; max-width:208px !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding:1rem .7rem 7rem; }
/* permanently hide sidebar collapse / expand controls */
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapseButton"] button,
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] {
  display:none !important; }
section[data-testid="stSidebar"] .stButton button { justify-content:flex-start; gap:.6rem; text-align:left;
  border:1px solid transparent; background:transparent; color:var(--muted); border-radius:12px;
  padding:.6rem .8rem; font-weight:500; box-shadow:none; }
section[data-testid="stSidebar"] .stButton button:hover { background:var(--surface2); color:var(--text); }
section[data-testid="stSidebar"] .stButton button[kind="primary"],
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] {
  background:var(--accent-soft); color:var(--accent3); border-color:rgba(255,122,26,.35); }
.st-key-sb_bottom { position:fixed; bottom:14px; left:12px; width:184px; z-index:5; }
section[data-testid="stSidebar"][aria-expanded="false"] .st-key-sb_bottom { display:none; }
.sb-profile { display:flex; align-items:center; gap:.65rem; padding:.7rem .75rem; margin-top:.7rem;
  border:1px solid var(--border); border-radius:14px; background:var(--surface2); }
.avatar { width:34px; height:34px; border-radius:50%; background:var(--accent-soft); color:var(--accent2);
  display:flex; align-items:center; justify-content:center; font-weight:600; font-size:.8rem; flex:none; }
.sb-name { font-weight:600; font-size:.88rem; line-height:1.2; }
.sb-role { color:var(--muted); font-size:.75rem; line-height:1.2; }

/* ---------- page header ---------- */
.page-title { font-size:1.7rem; font-weight:700; letter-spacing:-.03em; line-height:1.15; }
.page-sub { color:var(--muted); font-size:.92rem; margin-top:.25rem; }
.cust-title { font-size:1.5rem; font-weight:700; letter-spacing:-.02em; color:var(--accent3); }
.st-key-prep_btn button { min-height:46px; border-radius:12px; font-weight:600; }
.stButton button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] button[kind="primary"] {
  background:linear-gradient(135deg,#ff7a1a,#ff9f45); color:#111; border:none; font-weight:600; }
.stButton button[kind="primary"]:hover { filter:brightness(1.08); }

/* ---------- info cards ---------- */
.st-key-card_customer, .st-key-card_product, .st-key-card_memory {
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  padding:1.15rem 1.3rem 1rem; box-shadow:var(--shadow); min-height:212px; }
.card-label { font-size:.68rem; font-weight:600; letter-spacing:.09em; color:var(--muted); text-transform:uppercase; }
.card-value { font-size:1.35rem; font-weight:700; letter-spacing:-.02em; margin:.45rem 0 .7rem; line-height:1.2; }
.card-sub { color:var(--muted); font-size:.85rem; line-height:1.45; }
.meta-row { display:flex; justify-content:space-between; font-size:.85rem; padding:.28rem 0; border-top:1px solid var(--border); }
.meta-row span { color:var(--muted); } .meta-row b { font-weight:600; }
.prod-icon { width:38px; height:38px; border-radius:11px; background:var(--accent-soft); display:flex;
  align-items:center; justify-content:center; font-size:1.15rem; flex:none; }
.prod-row { display:flex; gap:.75rem; align-items:center; margin:.45rem 0 .55rem; }
.prod-row .card-value { margin:0; }
.status { display:inline-flex; align-items:center; gap:.5rem; font-weight:700; font-size:1.2rem; margin:.45rem 0 .6rem; }
.dot { width:9px; height:9px; border-radius:50%; display:inline-block; background:var(--muted); }
.dot.green { background:var(--green); box-shadow:0 0 0 4px rgba(255,178,94,.18); }
.dot.amber { background:var(--amber); box-shadow:0 0 0 4px rgba(255,122,26,.18); }
.dot.red { background:var(--red); box-shadow:0 0 0 4px rgba(255,77,46,.18); }
[class*="st-key-card_"] .stButton button, [class*="st-key-card_"] [data-testid="stPopover"] button {
  background:transparent; border:none; color:var(--accent2); padding:.25rem 0; font-weight:600;
  font-size:.85rem; justify-content:flex-start; box-shadow:none; min-height:0; }
[class*="st-key-card_"] .stButton button:hover, [class*="st-key-card_"] [data-testid="stPopover"] button:hover {
  background:transparent; color:var(--accent3); }

/* ---------- memory flow strip ---------- */
.flow { display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; margin:1.1rem 0 .2rem; color:var(--muted); font-size:.82rem; }
.flow .step { display:flex; align-items:center; gap:.5rem; background:var(--surface); border:1px solid var(--border);
  border-radius:999px; padding:.38rem .85rem .38rem .45rem; }
.flow .step b { width:20px; height:20px; border-radius:50%; background:var(--accent-soft); color:var(--accent2);
  display:flex; align-items:center; justify-content:center; font-size:.7rem; }
.flow .arrow { color:var(--border); }

/* ---------- assistant area & chips ---------- */
.assist { display:flex; align-items:center; gap:.55rem; margin:1.4rem 0 .6rem; font-size:1.05rem; font-weight:600; }
.assist i { color:var(--accent); font-style:normal; }
.st-key-chips .stButton button { border-radius:999px; font-size:.8rem; font-weight:500; padding:.38rem .9rem;
  background:var(--surface); border:1px solid var(--border); color:var(--text); min-height:0; box-shadow:none; }
.st-key-chips .stButton button:hover { border-color:var(--accent); color:#fff; background:var(--accent-soft); }

/* ---------- pills ---------- */
.pill { display:inline-block; font-size:.78rem; font-weight:500; padding:.22rem .7rem; border-radius:999px;
  background:var(--surface2); border:1px solid var(--border); margin:0 .35rem .35rem 0; }
.pill.obj { color:#ff9a70; border-color:rgba(255,120,70,.35); background:rgba(255,120,70,.09); }
.pill.comp { color:#ffc078; border-color:rgba(255,192,120,.35); background:rgba(255,192,120,.09); }
.pill.need { color:#ffd9a8; border-color:rgba(255,217,168,.3); background:rgba(255,217,168,.07); }
.pill.next { color:#ffb27a; border-color:rgba(255,122,26,.4); background:var(--accent-soft); }
.tag { display:inline-block; font-size:.68rem; font-weight:700; letter-spacing:.05em; padding:.18rem .6rem;
  border-radius:999px; margin-right:.4rem; background:var(--surface2); color:var(--muted); border:1px solid var(--border); }
.tag.PRICING { color:#ffb27a; } .tag.COMPETITOR { color:#ff9a70; } .tag.STAKEHOLDER { color:#ffd9a8; }
.tag.PRODUCT { color:#ffc078; } .tag.PROFILE { color:#ffa64d; }

/* ---------- confirmation / saved / briefing ---------- */
[class*="st-key-pending_"] { background:var(--surface); border:1px solid rgba(255,122,26,.4); border-radius:var(--radius);
  padding:1.1rem 1.3rem; box-shadow:var(--shadow); margin:1rem 0; }
.pending-head { font-weight:700; margin-bottom:.15rem; }
.pending-quote { color:var(--muted); font-size:.88rem; margin:.3rem 0 .8rem; border-left:2px solid var(--accent); padding-left:.75rem; }
.field-label { font-size:.7rem; font-weight:600; letter-spacing:.07em; color:var(--muted); text-transform:uppercase; margin:.55rem 0 .3rem; }
.saved { background:rgba(255,178,94,.07); border:1px solid rgba(255,178,94,.35); border-radius:var(--radius); padding:.9rem 1.15rem; margin:.9rem 0; }
.saved.warn { background:rgba(255,77,46,.07); border-color:rgba(255,77,46,.4); }
.saved-title { font-weight:700; margin-bottom:.3rem; } .saved-row { font-size:.88rem; color:var(--muted); }
.saved-row b { color:var(--text); font-weight:600; }
.briefing { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  padding:1.2rem 1.4rem; box-shadow:var(--shadow); max-width:640px; }
.briefing .b-label { font-size:.68rem; font-weight:700; letter-spacing:.12em; color:var(--accent); }
.briefing .b-cust { font-size:1.2rem; font-weight:700; margin:.25rem 0 .1rem; }
.briefing .b-head { color:var(--muted); font-size:.9rem; margin-bottom:.5rem; }
.briefing h5 { font-size:.72rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); margin:1rem 0 .35rem; }
.briefing ul, .briefing ol { margin:0; padding-left:1.15rem; } .briefing li { margin:.22rem 0; font-size:.93rem; }

/* ---------- memory timeline ---------- */
.tl { position:relative; margin:.4rem 0 1rem; padding-left:1.6rem; }
.tl:before { content:""; position:absolute; left:.42rem; top:.5rem; bottom:.5rem; width:2px; background:var(--border); }
.tl-item { position:relative; padding:0 0 1.15rem; }
.tl-item:before { content:""; position:absolute; left:-1.42rem; top:.38rem; width:10px; height:10px; border-radius:50%;
  background:var(--accent); box-shadow:0 0 0 4px var(--accent-soft); }
.tl-date { color:var(--muted); font-size:.78rem; font-weight:600; letter-spacing:.03em; }
.tl-title { font-weight:600; margin:.1rem 0; } .tl-sum { color:var(--muted); font-size:.9rem; }
.tl-warn { color:var(--amber); font-size:.75rem; margin-left:.4rem; }
.mem-panel { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:1rem 1.2rem; min-height:110px; }
.empty { text-align:center; padding:2.2rem 1rem; color:var(--muted); background:var(--surface);
  border:1px dashed var(--border); border-radius:var(--radius); margin:1rem 0; }
.empty b { color:var(--text); display:block; font-size:1.05rem; margin-bottom:.3rem; }

/* ---------- auth ---------- */
.auth-brand { font-size:3rem; margin:2rem 0 .3rem; }
.auth-tag { text-align:center; color:var(--muted); margin-bottom:1.4rem; }
[data-testid="stForm"] { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:1.3rem; }

/* ---------- chat & bottom bar ---------- */
[data-testid="stChatMessage"] { background:transparent; padding:.5rem 0; }
[data-testid="stChatInput"] { border:1px solid var(--border); border-radius:18px; background:var(--surface); box-shadow:var(--shadow); }
[data-testid="stChatInput"]:focus-within { border-color:var(--accent); box-shadow:0 0 0 3px rgba(255,122,26,.18); }
[data-testid="stChatInput"] textarea { min-height:54px; }
[data-testid="stBottom"] > div { background:var(--bg); }
[data-testid="stBottomBlockContainer"] { max-width:1120px; padding-left:2rem; padding-right:2rem; }
[data-testid="stExpander"] { border:1px solid var(--border); border-radius:12px; background:var(--surface); }
.st-key-voice_fallback { position:fixed; bottom:92px; right:32px; z-index:20; }

/* ---------- responsive ---------- */
@media (max-width:1024px){
  .block-container { padding:1rem 1.2rem 8rem; }
  [data-testid="stBottomBlockContainer"] { padding-left:1.2rem; padding-right:1.2rem; }
}
@media (max-width:768px){
  .block-container { padding:.8rem .9rem 8rem; }
  .brand { font-size:1.8rem; } .auth-brand { font-size:2.2rem; } .page-title { font-size:1.35rem; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# =====================================================================
# 12. REUSABLE UI PIECES
# =====================================================================
def pills(values: list[str], cls: str = "") -> str:
    return "".join(f'<span class="pill {cls}">{esc(v)}</span>' for v in values) or '<span class="card-sub">—</span>'


def _voice_fallback(key: str):
    """Mic for Streamlit versions whose chat_input has no built-in audio."""
    with st.container(key="voice_fallback"):
        with st.popover("", icon=":material/mic:", help="Record a voice message"):
            clip_ = st.audio_input("Record, then stop", key=f"{key}_aud")
    if clip_ is not None:
        h = hashlib.md5(clip_.getvalue()).hexdigest()
        if st.session_state.get(f"{key}_last") != h:  # process each recording only once
            st.session_state[f"{key}_last"] = h
            return {"text": "", "files": [], "audio": clip_}
    return None


def composer(placeholder: str, key: str, files: bool = True):
    """Bottom bar: text, + attachments, microphone, send."""
    kw = {}
    if files and CI_FILES:
        kw.update(accept_file="multiple", file_type=FILE_TYPES)
    if CI_AUDIO:
        kw["accept_audio"] = True
    voice = None if CI_AUDIO else _voice_fallback(key)
    val = st.chat_input(placeholder, key=key, **kw)
    if val is None:
        return voice
    if isinstance(val, str):
        return {"text": val, "files": [], "audio": None}
    get = (lambda k: val.get(k)) if hasattr(val, "get") else (lambda k: getattr(val, k, None))
    return {"text": get("text") or "", "files": list(get("files") or []), "audio": get("audio")}


def _select_customer_cb(prefix: str) -> None:
    name = st.session_state.get(f"{prefix}_sel")
    if name in ws()["customers"]:
        ws()["selected_customer"] = name
        save_ws()


def _add_customer_cb(prefix: str) -> None:
    if create_customer(st.session_state.get(f"{prefix}_new", "")):
        st.session_state[f"{prefix}_new"] = ""


def customer_switcher(prefix: str) -> None:
    names = list(ws()["customers"].keys())
    sel = selected_customer()
    if names:
        if sel in names:  # keep every switcher in sync with the workspace's selected customer
            st.session_state[f"{prefix}_sel"] = sel
        st.selectbox("Select customer", names, key=f"{prefix}_sel",
                     on_change=_select_customer_cb, args=(prefix,))
    st.text_input("Add a new customer", key=f"{prefix}_new", placeholder="e.g. ABC Technologies")
    st.button("Add customer", key=f"{prefix}_add", on_click=_add_customer_cb, args=(prefix,), **STRETCH_BTN)


def render_flow_strip() -> None:
    st.markdown(
        '<div class="flow"><div class="step"><b>1</b>Daily update</div><span class="arrow">→</span>'
        '<div class="step"><b>2</b>AI understands it</div><span class="arrow">→</span>'
        '<div class="step"><b>3</b>Saved to customer memory</div><span class="arrow">→</span>'
        '<div class="step"><b>4</b>Used in every future answer</div></div>', unsafe_allow_html=True)


def render_cards() -> None:
    cust = selected_customer()
    snap = customer_snapshot(cust)
    product = ws().get("product")
    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(key="card_customer"):
            st.markdown(
                '<div class="card-label">Customer</div>'
                f'<div class="card-value">{esc(cust or "No customer selected")}</div>'
                f'<div class="meta-row"><span>Deal stage</span><b>{esc(snap["stage"] or "—")}</b></div>'
                f'<div class="meta-row"><span>Interest</span><b>{esc(snap["interest"] or "—")}</b></div>',
                unsafe_allow_html=True)
            with st.popover("Switch customer  ›", **STRETCH_POP):
                customer_switcher("home")

    with c2:
        with st.container(key="card_product"):
            name = product["name"] if product else "No product yet"
            sub = clip(product.get("summary", ""), 120) if product else "Upload your product document in Product Info."
            st.markdown(
                '<div class="card-label">Product</div>'
                f'<div class="prod-row"><div class="prod-icon">📦</div><div class="card-value">{esc(name)}</div></div>'
                f'<div class="card-sub">{esc(sub)}</div>', unsafe_allow_html=True)
            if st.button("Manage product  ›", key="card_product_btn", **STRETCH_BTN):
                go("product")

    with c3:
        with st.container(key="card_memory"):
            active = snap["count"] > 0
            hs_label, hs_cls = hs_state_label()
            dot = ("amber" if snap["unsynced"] else "green") if active else ""
            st.markdown(
                '<div class="card-label">Memory Status</div>'
                f'<div class="status"><span class="dot {dot}"></span>{"Memory Active" if active else "No memory yet"}</div>'
                f'<div class="meta-row"><span>Interactions remembered</span><b>{snap["count"]}</b></div>'
                f'<div class="meta-row"><span>Last updated</span><b>{esc(rel_day(snap["last"]))}</b></div>'
                f'<div class="meta-row"><span>Hindsight</span><b><span class="dot {hs_cls}" style="width:7px;height:7px"></span> {hs_label}</b></div>',
                unsafe_allow_html=True)
            if st.button("View memory  ›", key="card_memory_btn", **STRETCH_BTN):
                go("memory")


def render_saved_banner() -> None:
    s = st.session_state.get("last_saved")
    if not s:
        return
    rec = s["rec"]
    rows = []
    for label, key in (("Objection", "objections"), ("Competitor", "competitors"), ("Need", "needs")):
        for v in rec.get(key, []):
            rows.append(f'<div class="saved-row">{label}: <b>{esc(v)}</b></div>')
    if rec.get("next_action"):
        rows.append(f'<div class="saved-row">Next action: <b>{esc(rec["next_action"])}</b></div>')
    if rec.get("stage"):
        rows.append(f'<div class="saved-row">Deal stage: <b>{esc(rec["stage"])}</b></div>')
    if s["ok"]:
        head, cls = f'✓ Added to {esc(s["customer"])} memory', "saved"
    else:
        head, cls = f'⚠ Saved locally for {esc(s["customer"])} — Hindsight sync failed', "saved warn"
        rows.append(f'<div class="saved-row">{esc(clip(s["err"] or "", 200))} (use “Retry sync” in View memory)</div>')
    st.markdown(f'<div class="{cls}"><div class="saved-title">{head}</div>{"".join(rows)}</div>', unsafe_allow_html=True)


def render_briefing(data: dict, customer: str) -> None:
    if "raw" in data:
        st.markdown(data["raw"])
        return

    def li(items):
        return "".join(f"<li>{esc(i)}</li>" for i in items if str(i).strip())

    parts = [f'<div class="b-label">AI CALL BRIEFING</div><div class="b-cust">{esc(customer)}</div>']
    if data.get("headline"):
        parts.append(f'<div class="b-head">{esc(data["headline"])}</div>')
    if data.get("highlights"):
        parts.append(f'<ul>{li(data["highlights"])}</ul>')
    if data.get("discussion"):
        parts.append(f'<h5>Suggested discussion</h5><ol>{li(data["discussion"])}</ol>')
    if data.get("watch_outs"):
        parts.append(f'<h5>Watch out for</h5><ul>{li(data["watch_outs"])}</ul>')
    st.markdown(f'<div class="briefing">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"🧠 Memory used ({len(sources)})"):
        for s in sources:
            st.markdown(f'<span class="tag {categorize_memory(s["text"], s.get("scope", ""))}">'
                        f'{categorize_memory(s["text"], s.get("scope", ""))}</span> {esc(clip(s["text"], 400))}',
                        unsafe_allow_html=True)


def render_messages() -> None:
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"], avatar="🧠" if m["role"] == "assistant" else "🧑‍💼"):
            kind = m.get("kind", "text")
            if kind == "briefing":
                render_briefing(m["data"], m.get("customer", ""))
            elif kind == "error":
                st.error(m["content"])
            else:
                st.markdown(m["content"])
            for w in m.get("warns", []):
                st.caption(f"⚠️ {w}")
            render_sources(m.get("sources"))


def render_pending(default_customer: str | None) -> None:
    """Step 6 of the flow: user confirms what the AI extracted before it is saved."""
    p = st.session_state.get("pending_update")
    if not p:
        return
    d, pid = p["data"], p["id"]
    with st.container(key=f"pending_{pid}"):
        st.markdown('<div class="pending-head">🧠 Here’s what I understood — confirm to save it to memory</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="pending-quote">{esc(clip(p["raw"], 320))}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card-sub"><b style="color:var(--text)">{esc(d["title"])}</b> — {esc(d["summary"])}</div>'
            + ("<div class='field-label'>Needs</div>" + pills(d["needs"], "need") if d["needs"] else "")
            + ("<div class='field-label'>Objections</div>" + pills(d["objections"], "obj") if d["objections"] else "")
            + ("<div class='field-label'>Competitors</div>" + pills(d["competitors"], "comp") if d["competitors"] else "")
            + ("<div class='field-label'>Next action</div>" + pills([d["next_action"]], "next") if d["next_action"] else ""),
            unsafe_allow_html=True)
        with st.expander("Edit details"):
            cust = st.text_input("Customer", value=p["customer"], key=f"pu_cust_{pid}")
            title = st.text_input("Title", value=d["title"], key=f"pu_title_{pid}")
            summary = st.text_area("Summary", value=d["summary"], key=f"pu_sum_{pid}", height=80)
            e1, e2, e3 = st.columns(3)
            needs = e1.text_input("Needs (comma separated)", value=", ".join(d["needs"]), key=f"pu_needs_{pid}")
            objs = e2.text_input("Objections", value=", ".join(d["objections"]), key=f"pu_objs_{pid}")
            comps = e3.text_input("Competitors", value=", ".join(d["competitors"]), key=f"pu_comps_{pid}")
            nxt = st.text_input("Next action", value=d["next_action"], key=f"pu_next_{pid}")
            f1, f2, f3 = st.columns(3)
            stage = f1.selectbox("Deal stage", [""] + STAGES, index=([""] + STAGES).index(d["stage"]),
                                 format_func=lambda v: v or "—", key=f"pu_stage_{pid}")
            interest = f2.selectbox("Interest", [""] + INTERESTS, index=([""] + INTERESTS).index(d["interest"]),
                                    format_func=lambda v: v or "—", key=f"pu_int_{pid}")
            when = f3.date_input("Date", value=date.fromisoformat(d["date"]), max_value=date.today(), key=f"pu_date_{pid}")
        b1, b2, _ = st.columns([2, 1, 3])
        if b1.button("Confirm & save to memory", type="primary", key=f"pu_ok_{pid}", **STRETCH_BTN):
            if not cust.strip():
                st.warning("Add a customer name first (open “Edit details”).")
            else:
                rec = {"id": short_id(), "date": when.isoformat(), "ts": now_iso(), "title": title.strip() or "Daily update",
                       "summary": summary.strip(), "raw": p["text"], "needs": clean_list(needs),
                       "objections": clean_list(objs), "competitors": clean_list(comps),
                       "next_action": nxt.strip(), "stage": stage, "interest": interest, "source": p["source"]}
                if commit_update(cust, rec):
                    st.session_state.pending_update = None
                    st.session_state.chat_messages.append({
                        "role": "assistant", "kind": "text",
                        "content": f"✓ Saved to **{selected_customer()}** memory. Ask me anything about this deal, or tap “Prepare for my next call”."})
                    save_chat()
                    st.rerun()
        if b2.button("Discard", key=f"pu_no_{pid}", **STRETCH_BTN):
            st.session_state.pending_update = None
            st.rerun()


# =====================================================================
# 13. INPUT HANDLING (typed / voice / attachment) -> update or question
# =====================================================================
def read_submission(sub: dict):
    """Turn a composer submission into (display_text, full_text, source)."""
    typed = (sub["text"] or "").strip()
    source, extra = "text", []
    if sub.get("audio") is not None:
        try:
            with st.spinner("Transcribing your voice…"):
                spoken = transcribe(sub["audio"])
            typed = f"{typed} {spoken}".strip()
            source = "voice"
        except Exception as e:
            st.error(f"Voice transcription failed: {e}")
    names = []
    for f in sub.get("files") or []:
        try:
            body = extract_text(f)
            extra.append(f"[Attached: {f.name}]\n{clip(body, 6000)}")
            names.append(f.name)
            source = "file" if source == "text" else source
        except Exception as e:
            st.error(f"Couldn't read {f.name}: {e}")
    display = typed + ("".join(f"\n\n📎 {n}" for n in names))
    full = (typed + ("\n\n" + "\n\n".join(extra) if extra else "")).strip()
    if source == "voice" and typed:
        display = "🎙️ " + display
    return display.strip(), full, source


def handle_submission(sub: dict, force_update: bool = False) -> None:
    display, full, source = read_submission(sub)
    if not full:
        return
    user = current_user()
    msgs = st.session_state.chat_messages
    st.session_state.last_saved = None
    msgs.append({"role": "user", "kind": "text", "content": display})
    with st.spinner("Understanding your message…"):
        data = analyze_message(full, user["profession"] or "sales professional", force_update)
    if data["intent"] == "update":
        known = resolve_customer(data["customer"]) if data["customer"] else ""
        st.session_state.pending_update = {
            "id": short_id(), "customer": known or selected_customer() or "", "raw": display,
            "text": full, "source": source, "data": data}
        msgs.append({"role": "assistant", "kind": "text",
                     "content": "That sounds like a **daily update**. Review what I extracted below and confirm to add it to customer memory."})
        save_chat()
    else:
        run_question(display, full, record_user=False)


def run_question(display: str, question: str, briefing: bool = False, record_user: bool = True) -> None:
    msgs = st.session_state.chat_messages
    cust = selected_customer()
    if record_user:
        msgs.append({"role": "user", "kind": "text", "content": display})
    if briefing and not cust:
        msgs.append({"role": "assistant", "kind": "error", "content": "Select or add a customer first, then I can prepare your briefing."})
        save_chat()
        return
    history = [m for m in msgs[:-1] if m.get("kind", "text") == "text"]
    try:
        with st.spinner("Recalling memory and thinking…"):
            if briefing:
                data, mem, warns = generate_briefing(cust)
                msgs.append({"role": "assistant", "kind": "briefing", "data": data, "customer": cust,
                             "sources": mem, "warns": warns, "content": "Call briefing"})
            else:
                text, mem, warns = answer_question(cust, question, history)
                msgs.append({"role": "assistant", "kind": "text", "content": text, "sources": mem, "warns": warns})
    except Exception as e:
        msgs.append({"role": "assistant", "kind": "error", "content": f"I couldn't generate a response: {e}"})
    save_chat()


# =====================================================================
# 14. AUTH PAGES: login / sign-up -> role selection
# =====================================================================
def page_auth() -> None:
    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        st.markdown('<div class="auth-brand">OutStride AI</div>'
                    '<div class="auth-tag">The sales assistant that never forgets a deal.</div>', unsafe_allow_html=True)
        tab_in, tab_up = st.tabs(["Log in", "Sign up"])
        with tab_in:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="you@company.com")
                pw = st.text_input("Password", type="password")
                go_in = st.form_submit_button("Log in", type="primary", **STRETCH_FORM)
            if go_in:
                ok, res = check_login(email, pw)
                if ok:
                    sign_in(res)
                    st.rerun()
                else:
                    st.error(res)
        with tab_up:
            with st.form("signup_form"):
                name = st.text_input("Full name")
                email2 = st.text_input("Work email", placeholder="you@company.com")
                company = st.text_input("Company (optional)")
                pw2 = st.text_input("Password (min 6 characters)", type="password")
                go_up = st.form_submit_button("Create account", type="primary", **STRETCH_FORM)
            if go_up:
                ok, res = create_account(name, email2, pw2, company)
                if ok:
                    sign_in(res)
                    with st.spinner("Creating your memory bank in Hindsight…"):
                        sync_profile(current_user(), "Account created.")
                    st.rerun()
                else:
                    st.error(res)


def sync_profile(user: dict, note: str) -> None:
    """Store the person's details in Hindsight (own bank + shared registry)."""
    text = f"User profile: {user['name']} ({user['email']})"
    text += f" works as a {user['profession']}" if user.get("profession") else " uses OutStride AI"
    text += f" at {user['company']}." if user.get("company") else "."
    text += f" {note} Account created {user['created'][:10]}."
    for bank, name in ((user["bank_id"], f"OutStride — {user['name']}"), (REGISTRY_BANK, "OutStride users")):
        try:
            hs_retain(bank, text, context="User profile", ts=now_iso(), bank_name=name)
        except Exception as e:
            st.toast(f"Couldn't sync profile to Hindsight: {clip(str(e), 80)}", icon="⚠️")
            break


def page_profession() -> None:
    user = current_user()
    editing = st.session_state.editing_role
    if st.session_state.role_choice is None:
        known = [r[0] for r in ROLES]
        cur = user.get("profession", "")
        st.session_state.role_choice = cur if cur in known else ("Other" if cur else None)
    choice = st.session_state.role_choice

    _, mid, _ = st.columns([0.5, 3, 0.5])
    with mid:
        st.markdown(f'<div class="auth-brand" style="margin-top:1.2rem;font-size:2.2rem">Welcome, {esc(user["name"].split()[0])} 👋</div>'
                    '<div class="auth-tag">What best describes your role? OutStride tailors its coaching to it.</div>',
                    unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (title, desc) in enumerate(ROLES):
            with cols[i % 3]:
                if st.button(title, key=f"role_{i}", type="primary" if choice == title else "secondary", **STRETCH_BTN):
                    st.session_state.role_choice = title
                    st.rerun()
                st.caption(desc)
        other = ""
        if choice == "Other":
            other = st.text_input("Your role", placeholder="e.g. Partnerships lead", key="role_other")
        c1, c2, _ = st.columns([1.3, 1, 2])
        can_go = bool(choice) and (choice != "Other" or other.strip())
        if c1.button("Continue", type="primary", disabled=not can_go, key="role_continue", **STRETCH_BTN):
            profession = other.strip() if choice == "Other" else choice
            update_user(user["email"], profession=profession)
            with st.spinner("Saving your profile to Hindsight…"):
                sync_profile(current_user(), "Role selected." if not editing else "Role updated.")
            st.session_state.role_choice = None
            st.session_state.editing_role = False
            st.session_state.page = "home"
            st.rerun()
        if editing and c2.button("Cancel", key="role_cancel", **STRETCH_BTN):
            st.session_state.role_choice = None
            st.session_state.editing_role = False
            st.session_state.page = "home"
            st.rerun()


# =====================================================================
# 15. APP SHELL: top bar + left sidebar
# =====================================================================
def render_topbar() -> None:
    user = current_user()
    with st.container(key="topbar"):
        _, c2, c3 = st.columns([1, 3, 1])
        with c2:
            st.markdown('<div class="brand">OutStride AI</div>', unsafe_allow_html=True)
        with c3:
            with st.container(key="topbar_right"):
                with st.popover(user["name"].split()[0], icon=":material/account_circle:"):
                    st.markdown(
                        f'<div class="sb-profile" style="margin:0 0 .6rem"><div class="avatar">{esc(initials(user["name"]))}</div>'
                        f'<div><div class="sb-name">{esc(user["name"])}</div><div class="sb-role">{esc(user["profession"] or "—")}</div></div></div>'
                        f'<div class="meta-row"><span>Email</span><b>{esc(user["email"])}</b></div>'
                        f'<div class="meta-row"><span>Company</span><b>{esc(user.get("company") or "—")}</b></div>'
                        f'<div class="meta-row"><span>Member since</span><b>{esc(fmt_day(user["created"][:10]))}</b></div>'
                        f'<div class="meta-row"><span>Memory bank</span><b style="font-size:.75rem">{esc(user["bank_id"])}</b></div>',
                        unsafe_allow_html=True)
                    if st.button("Change role", key="tb_role", **STRETCH_BTN):
                        st.session_state.editing_role = True
                        st.session_state.role_choice = None
                        go("profession")
                    if st.button("Log out", key="tb_logout", **STRETCH_BTN):
                        sign_out()
                        st.rerun()


def render_sidebar() -> None:
    user = current_user()
    nav = [("home", "New Chat", ":material/add_comment:"),
           ("retain", "Retain", ":material/edit_note:"),
           ("recall", "Recall", ":material/manage_search:"),
           ("chatspace", "Chat Space", ":material/forum:")]
    page = st.session_state.page
    with st.sidebar:
        for key, label, icon in nav:
            active = page == key or (key == "home" and page in ("memory", "product"))
            if st.button(label, key=f"nav_{key}", icon=icon, type="primary" if active else "secondary", **STRETCH_BTN):
                if key == "home":
                    new_chat()
                go(key)
        with st.container(key="sb_bottom"):
            st.markdown(
                f'<div class="sb-profile"><div class="avatar">{esc(initials(user["name"]))}</div>'
                f'<div><div class="sb-name">{esc(user["name"])}</div><div class="sb-role">{esc(user["profession"] or "—")}</div></div></div>',
                unsafe_allow_html=True)


# =====================================================================
# 16. PAGES
# =====================================================================
CHIPS = [
    ("Summarize Deal", "Summarize this deal: current stage, customer interest, key needs, objections, competitors and the next action.", False),
    ("What happened before?", "Walk me through the history of this deal in chronological order.", False),
    ("Prepare me for the next call", "Prepare me for my next call", True),
    ("Show objections", "List every objection raised so far, how each one evolved, and how I should address it.", False),
    ("Create follow-up", "Draft a concise follow-up email to this customer based on the deal history and the agreed next step.", False),
]


def page_home() -> None:
    cust = selected_customer()
    sub = composer("Tell me what happened with this customer today...", "composer_home")

    h1, h2 = st.columns([3, 2])
    with h1:
        st.markdown(f'<div class="cust-title">{esc(cust) if cust else "Select a customer to begin"}</div>',
                    unsafe_allow_html=True)
    with h2:
        with st.container(key="prep_btn"):
            prep = st.button("Prepare for my next call", type="primary", icon=":material/bolt:",
                             key="prep_home", disabled=not cust, **STRETCH_BTN)
    st.write("")
    render_cards()

    if not ws()["customers"]:
        st.markdown('<div class="empty"><b>Add your first customer</b>Use “Switch customer” above, the + button, or load a ready-made demo deal.</div>',
                    unsafe_allow_html=True)
        if st.button("Load demo deal", key="home_demo"):
            load_demo_deal()
            st.rerun()

    st.markdown(f'<div class="assist"><i>✦</i>How can I help with {"<b>" + esc(cust) + "</b>" if cust else "this deal"}?</div>',
                unsafe_allow_html=True)
    clicked = None
    with st.container(key="chips"):
        cols = st.columns(len(CHIPS))
        for i, (label, prompt, is_brief) in enumerate(CHIPS):
            if cols[i].button(label, key=f"chip_{i}", disabled=not cust, **STRETCH_BTN):
                clicked = (label, prompt, is_brief)

    # --- process actions (state changes happen before the results are drawn) ---
    if prep:
        run_question("Prepare me for my next call", "", briefing=True)
    elif clicked:
        run_question(clicked[0], clicked[1], briefing=clicked[2])
    elif sub:
        handle_submission(sub)

    render_saved_banner()
    render_messages()
    render_pending(cust)


def page_retain() -> None:
    cust = selected_customer()
    sub = composer("Tell me what happened with this customer today...", "composer_retain")
    h1, h2 = st.columns([3, 2])
    with h1:
        st.markdown('<div class="page-title">Retain</div>'
                    '<div class="page-sub">Log today’s update in plain language — OutStride turns it into customer memory.</div>',
                    unsafe_allow_html=True)
    with h2:
        with st.popover(f"Customer: {cust or 'none'}  ›", **STRETCH_POP):
            customer_switcher("retain")
    render_flow_strip()
    st.markdown("<div class='page-sub' style='margin:.8rem 0'>Example: “Had a call with ABC today. They liked the product but pricing is still a concern. They are comparing us with XYZ.”</div>",
                unsafe_allow_html=True)
    if sub:
        handle_submission(sub, force_update=True)
    render_saved_banner()
    render_messages()
    render_pending(cust)

    snap = customer_snapshot(cust)
    if snap["count"]:
        st.markdown('<div class="field-label" style="margin-top:1.4rem">Recent updates</div>', unsafe_allow_html=True)
        for r in reversed(snap["interactions"][-4:]):
            st.markdown(f'<div class="mem-panel" style="min-height:0;margin-bottom:.6rem"><div class="tl-date">{esc(fmt_day(r["date"]))}</div>'
                        f'<div class="tl-title">{esc(r["title"])}</div><div class="tl-sum">{esc(r["summary"])}</div></div>',
                        unsafe_allow_html=True)


def run_recall(query: str, scope: str) -> None:
    cust = selected_customer() if scope == "customer" else None
    with st.spinner("Querying Hindsight memory layer..."):
        try:
            text, mem, warns = answer_question(cust, query)
            st.session_state.recall_state = {"query": query, "scope": scope, "answer": text, "items": mem, "warns": warns}
        except Exception as e:
            st.session_state.recall_state = {"query": query, "scope": scope, "answer": "", "items": [], "warns": [str(e)]}


def page_recall() -> None:
    cust = selected_customer()
    sub = composer("Ask your memory… e.g. What objections has ABC raised?", "composer_recall", files=False)
    st.markdown('<div class="page-title">Recall</div>'
                '<div class="page-sub">Draw back anything you’ve told OutStride — pulled straight from Hindsight memory.</div>',
                unsafe_allow_html=True)
    scope_label = st.radio("Search in", ["Selected customer", "All memory"], horizontal=True, key="recall_scope",
                           label_visibility="collapsed")
    scope = "customer" if scope_label == "Selected customer" and cust else "all"
    if scope_label == "Selected customer" and not cust:
        st.caption("No customer selected — searching all memory.")

    quick = [("Everything about this customer", "Give me everything you remember about this customer."),
             ("Objections so far", "What objections has this customer raised?"),
             ("Competitors mentioned", "Which competitors were mentioned and what was said about them?"),
             ("Recent activity", "What are the most recent updates?")]
    with st.container(key="chips"):
        cols = st.columns(len(quick))
        for i, (label, q) in enumerate(quick):
            if cols[i].button(label, key=f"rq_{i}", **STRETCH_BTN):
                run_recall(q, scope)
    if sub:
        text = (sub["text"] or "").strip()
        if sub.get("audio") is not None:
            try:
                with st.spinner("Transcribing your voice…"):
                    text = f"{text} {transcribe(sub['audio'])}".strip()
            except Exception as e:
                st.error(f"Voice transcription failed: {e}")
        if text:
            run_recall(text, scope)

    rs = st.session_state.recall_state
    if not rs:
        st.markdown('<div class="empty"><b>Ask your memory anything</b>Try “What did ABC say about pricing?” or tap a suggestion above.</div>',
                    unsafe_allow_html=True)
        return
    st.markdown(f'<div class="field-label" style="margin-top:1rem">Question</div><div>{esc(rs["query"])}</div>', unsafe_allow_html=True)
    for w in rs["warns"]:
        st.warning(w)
    if rs["answer"]:
        st.markdown('<div class="field-label">Answer from memory</div>', unsafe_allow_html=True)
        st.markdown(rs["answer"])
    st.markdown(f'<div class="field-label" style="margin-top:1.2rem">Raw memory from Hindsight ({len(rs["items"])})</div>',
                unsafe_allow_html=True)
    if not rs["items"]:
        st.caption("Hindsight returned nothing for this query.")
    for i, it in enumerate(rs["items"], 1):
        cat = categorize_memory(it["text"], it.get("scope", ""))
        with st.expander(f"Memory #{i} — {clip(it['text'], 70)}"):
            meta = " · ".join(x for x in (it.get("type"), it.get("when")) if x)
            st.markdown(f'<span class="tag {cat}">{cat}</span> <span class="card-sub">{esc(meta)}</span>', unsafe_allow_html=True)
            st.write(it["text"])


def page_product() -> None:
    product = ws().get("product")
    if st.button("← Back to dashboard", key="prod_back"):
        go("home")
    st.markdown('<div class="page-title">Product Info</div>'
                '<div class="page-sub">Upload your product documents once — OutStride stores them in Hindsight and uses them in every answer.</div>',
                unsafe_allow_html=True)
    st.write("")
    if product:
        files_html = "".join(f'<div class="meta-row"><span>📄 {esc(f["name"])}</span><b>{f["stored"]}/{f["chunks"]} parts stored</b></div>'
                             for f in product.get("files", [])) or '<div class="card-sub">No documents uploaded yet.</div>'
        st.markdown(f'<div class="mem-panel"><div class="card-label">Current product</div>'
                    f'<div class="prod-row"><div class="prod-icon">📦</div><div class="card-value">{esc(product["name"])}</div></div>'
                    f'<div class="card-sub" style="margin-bottom:.7rem">{esc(product.get("summary", ""))}</div>{files_html}</div>',
                    unsafe_allow_html=True)
        st.write("")
    name = st.text_input("Product / service name", value=(product or {}).get("name", ""), key="prod_name",
                         placeholder="e.g. AI Automation Platform")
    files = st.file_uploader("Product documents (PDF, DOCX, TXT, MD)", type=["pdf", "docx", "txt", "md"],
                             accept_multiple_files=True, key="prod_files")
    if st.button("Save to Hindsight memory", type="primary", disabled=not (name.strip() and files), key="prod_save"):
        ingest_product(name.strip(), files)
        st.rerun()
    st.caption("Large documents are split into parts. Once stored, product knowledge is recalled automatically when you ask questions.")


def page_chatspace() -> None:
    st.markdown('<div class="page-title">Chat Space</div><div class="page-sub">Your saved conversations.</div>', unsafe_allow_html=True)
    st.write("")
    chats = sorted(ws()["chats"], key=lambda c: c.get("updated", ""), reverse=True)
    if not chats:
        st.markdown('<div class="empty"><b>No saved chats yet</b>Conversations from New Chat show up here automatically.</div>', unsafe_allow_html=True)
        return
    for c in chats:
        col1, col2, col3 = st.columns([6, 1.2, 1.2])
        with col1:
            st.markdown(f'<div class="mem-panel" style="min-height:0"><div class="tl-title">{esc(c["title"])}</div>'
                        f'<div class="tl-sum">{esc(c.get("customer") or "No customer")} · {esc(rel_day(c.get("updated", "")[:10]))} · '
                        f'{len(c["messages"])} messages</div></div>', unsafe_allow_html=True)
        if col2.button("Open", key=f"open_{c['id']}", **STRETCH_BTN):
            st.session_state.chat_id = c["id"]
            st.session_state.chat_messages = c["messages"]
            st.session_state.pending_update = None
            if c.get("customer") in ws()["customers"]:
                ws()["selected_customer"] = c["customer"]
                save_ws()
            go("home")
        if col3.button("Delete", key=f"del_{c['id']}", **STRETCH_BTN):
            ws()["chats"] = [x for x in ws()["chats"] if x["id"] != c["id"]]
            save_ws()
            st.rerun()


def page_memory() -> None:
    cust = selected_customer()
    b1, b2 = st.columns([3, 2])
    with b1:
        if st.button("← Back to dashboard", key="mem_back"):
            go("home")
    if not cust:
        st.markdown('<div class="empty"><b>No customer selected</b>Pick a customer on the dashboard first.</div>', unsafe_allow_html=True)
        return
    snap = customer_snapshot(cust)
    with b2:
        with st.container(key="prep_btn"):
            if st.button("Prepare for my next call", type="primary", icon=":material/bolt:", key="prep_mem", **STRETCH_BTN):
                run_question("Prepare me for my next call", "", briefing=True)
                go("home")
    st.markdown(f'<div class="page-title">{esc(cust)}</div><div class="page-sub">DEAL MEMORY · {snap["count"]} interactions</div>', unsafe_allow_html=True)
    st.write("")
    if not snap["count"]:
        st.markdown('<div class="empty"><b>No memory yet</b>Log your first update and it will appear here as a timeline.</div>', unsafe_allow_html=True)
        return
    if snap["unsynced"]:
        st.warning(f"{snap['unsynced']} update(s) are saved locally but not yet in Hindsight.")
        if st.button("Retry sync", key="mem_retry"):
            retry_sync(cust)
            st.rerun()
    st.markdown('<div class="field-label">Timeline</div>', unsafe_allow_html=True)
    items = "".join(
        f'<div class="tl-item"><div class="tl-date">{esc(fmt_day(r["date"]))}'
        f'{"<span class=tl-warn>not synced</span>" if not r.get("synced") else ""}</div>'
        f'<div class="tl-title">{esc(r["title"])}</div><div class="tl-sum">→ {esc(r["summary"])}</div></div>'
        for r in snap["interactions"])
    st.markdown(f'<div class="tl">{items}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, label, vals, cls in ((c1, "Key needs", snap["needs"], "need"), (c2, "Objections", snap["objections"], "obj"),
                                  (c3, "Competitors", snap["competitors"], "comp")):
        with col:
            st.markdown(f'<div class="mem-panel"><div class="card-label" style="margin-bottom:.6rem">{label}</div>{pills(vals, cls)}</div>',
                        unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="mem-panel" style="min-height:0"><div class="card-label" style="margin-bottom:.6rem">Next action</div>'
                f'{pills([snap["next_action"]], "next") if snap["next_action"] else "<span class=card-sub>—</span>"}</div>',
                unsafe_allow_html=True)


# =====================================================================
# 17. ROUTER
# =====================================================================
def main() -> None:
    init_state()
    inject_css()
    user = current_user()
    if not user:
        page_auth()
        return
    if st.session_state.ws is None:
        st.session_state.ws = load_ws(user["email"])
    if not user.get("profession") or st.session_state.page == "profession":
        page_profession()
        return
    render_topbar()
    render_sidebar()
    {"home": page_home, "retain": page_retain, "recall": page_recall, "product": page_product,
     "chatspace": page_chatspace, "memory": page_memory}.get(st.session_state.page, page_home)()


main()
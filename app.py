import difflib
import html
import os
import traceback

import streamlit as st
from pipeline import (transcribe, refine, generate_record, to_markdown, to_json, to_zip,
                      PipelineError, ALLOWED, STT_MODEL, REFINER_MODEL, MINUTES_MODEL)

st.set_page_config(page_title="Lexicon AI", page_icon="🎙️", layout="wide",
                   initial_sidebar_state="expanded")

try:  # also accept the key from .streamlit/secrets.toml (local) or Streamlit Cloud secrets
    os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    pass

ss = st.session_state
if "ui_uploader" not in ss:
    ss.ui_uploader = 0  # bumping this gives a fresh, empty uploader ("Process another file")

STEPS = [("Transcribing", STT_MODEL, "Speech becomes a raw transcript."),
         ("Refining", REFINER_MODEL, "Misheard technical terms are fixed."),
         ("Generating minutes", MINUTES_MODEL, "Minutes, decisions and action items are written.")]
STATE_TEXT = {"pending": "Waiting", "running": "In progress…", "done": "Done", "failed": "Failed"}
FORMATS = ", ".join(e.lstrip(".").upper() for e in sorted(ALLOWED))

# ---------------- styling (UI only) ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,600;0,700;0,800;1,600;1,700&family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root { --mt-grad:linear-gradient(90deg,#8b5cf6,#ec4899,#f97316); --mt-line:rgba(128,128,128,.22); }
/* body text: Inter (text elements only, so Streamlit's icon font is left alone) */
html, body, .stApp { font-family:'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp td, .stApp th,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
.stApp [data-baseweb="tab"] { font-family:'Inter', system-ui, sans-serif !important; }
/* fancy serif for the name, headings and numbers */
.mt-title, .mt-results-title, .mt-empty-title, .mt-stat-value, .mt-topic, .mt-dec-n,
.mt-how-n, .mt-how b, .mt-error-title, .mt-n {
  font-family:'Fraunces', Georgia, 'Times New Roman', serif !important; letter-spacing:-.015em; }
/* small uppercase labels and tabs: clean sans */
.mt-side-title, .mt-label, .mt-stat-label, .stApp [data-baseweb="tab"] p {
  font-family:'Plus Jakarta Sans', 'Inter', sans-serif !important; }
.stApp code, .stApp pre code, .mt-error-msg {
  font-family:'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important; }
.block-container, [data-testid="stMainBlockContainer"] { max-width:1100px; padding-top:2.2rem; }
.mt-c0 { --tone:#8b5cf6; } .mt-c1 { --tone:#ec4899; } .mt-c2 { --tone:#0ea5e9; }
.mt-c3 { --tone:#10b981; } .mt-c4 { --tone:#f97316; }
/* header */
.mt-title { font-size:3.1rem; font-weight:800; line-height:1.05; }
.mt-title em { font-style:italic; font-weight:700; padding-right:.1em; background:var(--mt-grad);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.mt-sub { font-size:1.1rem; opacity:.72; margin-top:.5rem; max-width:720px; line-height:1.55; }
.mt-rule { height:3px; width:72px; border-radius:3px; background:var(--mt-grad); margin:1.1rem 0 1.8rem; }
/* sidebar */
.mt-side-title { font-weight:800; font-size:.75rem; letter-spacing:.09em; text-transform:uppercase;
  opacity:.6; margin:1.4rem 0 .6rem; }
.mt-fileinfo { font-size:.9rem; margin:.1rem 0 .5rem; word-break:break-all; }
.mt-fileinfo span { opacity:.6; }
.mt-warn { color:#d97706; font-size:.86rem; font-weight:600; margin:.1rem 0 .6rem; }
.mt-st { position:relative; display:flex; gap:.75rem; padding-bottom:1.15rem; }
.mt-st:not(:last-child)::before { content:""; position:absolute; left:.78rem; top:1.75rem; bottom:.15rem;
  width:2px; background:var(--mt-line); }
.mt-st.mt-done:not(:last-child)::before { background:#10b981; }
.mt-st-dot { flex-shrink:0; width:1.6rem; height:1.6rem; border-radius:50%; display:inline-flex;
  align-items:center; justify-content:center; font-size:.78rem; font-weight:700;
  border:2px solid rgba(128,128,128,.45); }
.mt-st b { display:block; font-weight:600; line-height:1.3; }
.mt-st small { display:block; opacity:.6; font-size:.76rem; }
.mt-st-state { display:block; font-size:.78rem; font-weight:700; opacity:.55; }
.mt-running .mt-st-dot { border-color:#8b5cf6; color:#8b5cf6; animation:mt-pulse 1.2s infinite; }
.mt-running .mt-st-state { color:#8b5cf6; opacity:1; }
.mt-done .mt-st-dot { background:#10b981; border-color:#10b981; color:#fff; }
.mt-done .mt-st-state { color:#10b981; opacity:1; }
.mt-failed .mt-st-dot { background:#ef4444; border-color:#ef4444; color:#fff; }
.mt-failed .mt-st-state { color:#ef4444; opacity:1; }
@keyframes mt-pulse { 0%,100% { box-shadow:0 0 0 0 rgba(139,92,246,.5); }
                      50% { box-shadow:0 0 0 7px rgba(139,92,246,0); } }
/* buttons and expanders without boxes */
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
  width:100%; border-radius:999px; font-weight:600; }
button[kind="primary"], [data-testid="stBaseButton-primary"] { background:var(--mt-grad) !important;
  border:none !important; color:#fff !important; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { filter:brightness(1.08); }
button[kind="primary"]:disabled, [data-testid="stBaseButton-primary"]:disabled { opacity:.4; }
[data-testid="stDownloadButton"] button { background:transparent; border:1px solid var(--mt-line); }
[data-testid="stDownloadButton"] button:hover { border-color:#8b5cf6; color:#8b5cf6; }
[data-testid="stExpander"] details { border:none !important; }
[data-testid="stExpander"] summary { padding-left:0 !important; }
/* empty state */
.mt-empty { text-align:center; padding:2.5rem 1rem .5rem; }
.mt-empty-icon { font-size:3rem; line-height:1; }
.mt-empty-title { font-size:1.9rem; font-weight:700; margin:.8rem 0 .35rem; }
.mt-empty-text { opacity:.7; max-width:520px; margin:0 auto; line-height:1.6; }
.mt-how { display:grid; grid-template-columns:repeat(3, 1fr); gap:2.4rem; max-width:880px; margin:3rem auto 0; }
@media (max-width:800px) { .mt-how { grid-template-columns:1fr; } }
.mt-how-n { font-size:2.5rem; font-weight:700; font-style:italic; line-height:1; color:var(--tone); }
.mt-how b { display:block; font-size:1.15rem; font-weight:700; margin:.5rem 0 .25rem; }
.mt-how p { margin:0; opacity:.72; font-size:.92rem; line-height:1.55; }
.mt-how code { font-size:.76rem; }
/* results */
.mt-results { display:flex; align-items:baseline; flex-wrap:wrap; gap:.8rem; }
.mt-results-title { font-size:2rem; font-weight:700; font-style:italic; padding-right:.1em;
  background:var(--mt-grad); -webkit-background-clip:text; background-clip:text; color:transparent; }
.mt-results-file { opacity:.6; }
.mt-stats { display:flex; flex-wrap:wrap; gap:3rem; margin:1rem 0 1.6rem; }
.mt-stat-value { font-size:2.4rem; font-weight:700; line-height:1.1; color:var(--tone); }
.mt-stat-label { font-size:.74rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; opacity:.6; }
.mt-note { opacity:.75; font-size:.93rem; margin:.3rem 0 1rem; }
.stTabs [data-baseweb="tab-list"] { gap:1.6rem; }
.stTabs [data-baseweb="tab"] { padding:.55rem 0; background:transparent !important; }
.stTabs [data-baseweb="tab"] p { font-weight:600; opacity:.65; }
.stTabs [aria-selected="true"] p { opacity:1; color:#8b5cf6 !important; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--mt-grad) !important; height:3px !important; }
/* reading views */
.mt-doc { max-width:780px; font-size:1rem; line-height:1.8; }
.mt-doc p { margin:0 0 1.1rem; }
.mt-label { font-weight:800; font-size:.76rem; letter-spacing:.09em; text-transform:uppercase;
  color:var(--tone); margin:.6rem 0 .7rem; }
.mt-del { color:#ef4444; text-decoration:line-through; text-decoration-thickness:2px; }
.mt-ins { color:#10b981; font-weight:700; }
.mt-change { padding:.45rem 0; border-bottom:1px solid var(--mt-line); max-width:640px; }
.mt-change .mt-from { color:#ef4444; text-decoration:line-through; }
.mt-change .mt-arrow { opacity:.45; margin:0 .6rem; }
.mt-change .mt-to { color:#10b981; font-weight:700; }
.mt-lead { font-size:1.15rem; line-height:1.75; max-width:820px; margin:0 0 2rem; }
.mt-topic { font-weight:600; font-size:1.25rem; color:var(--tone); margin:1.5rem 0 .4rem; }
.mt-list { margin:0 0 .3rem 1.15rem; padding:0; max-width:820px; line-height:1.7; }
.mt-list li { margin-bottom:.3rem; }
.mt-dec { display:flex; gap:1.2rem; align-items:baseline; padding:1rem 0;
  border-bottom:1px solid var(--mt-line); max-width:880px; }
.mt-dec-n { font-size:1.8rem; font-weight:600; font-style:italic; color:var(--tone); min-width:2.6rem; }
.mt-dec-t { font-size:1.05rem; line-height:1.6; }
.mt-table { width:100%; border-collapse:collapse; }
.mt-table th { text-align:left; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em;
  opacity:.55; padding:.5rem 1rem .6rem 0; border-bottom:2px solid var(--mt-line); }
.mt-table td { padding:.95rem 1rem .95rem 0; border-bottom:1px solid var(--mt-line); vertical-align:top; line-height:1.5; }
.mt-table td.mt-n { font-size:1.15rem; font-weight:600; font-style:italic; color:var(--tone); width:2.6rem; }
.mt-owner { color:#8b5cf6; font-weight:600; }
.mt-deadline { color:#d97706; font-weight:600; }
.mt-unspec { opacity:.45; font-style:italic; white-space:nowrap; }
/* errors */
.mt-error { border-left:3px solid #ef4444; padding:.2rem 0 .2rem 1.1rem; margin:.5rem 0 1.5rem; }
.mt-error-title { color:#ef4444; font-weight:700; font-size:1.3rem; }
.mt-error-hint { opacity:.85; margin:.25rem 0 .5rem; }
.mt-error-msg { font-size:.85rem; opacity:.75; white-space:pre-wrap; word-break:break-word; }
</style>
""", unsafe_allow_html=True)


# ---------------- display helpers (UI only) ----------------
def esc(text) -> str:
    return html.escape(str(text))


def tone(i: int) -> str:
    return f"mt-c{i % 5}"


def note(text: str) -> None:
    st.markdown(f'<div class="mt-note">{text}</div>', unsafe_allow_html=True)


def fmt_size(n) -> str:
    size = float(n or 0)
    for unit in ("bytes", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "bytes" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def is_unspecified(value) -> bool:
    return str(value or "").strip().lower() in {"", "unspecified"}


def who(value) -> str:
    if is_unspecified(value):
        return '<span class="mt-unspec">Unspecified</span>'
    return f'<span class="mt-owner">👤 {esc(value)}</span>'


def when(value) -> str:
    if is_unspecified(value):
        return '<span class="mt-unspec">Unspecified</span>'
    return f'<span class="mt-deadline">📅 {esc(value)}</span>'


def stepper_html(states) -> str:
    rows = []
    for i, ((name, model, _), state) in enumerate(zip(STEPS, states)):
        mark = {"done": "✓", "failed": "✕"}.get(state, str(i + 1))
        rows.append(f'<div class="mt-st mt-{state}"><span class="mt-st-dot">{mark}</span><div>'
                    f'<b>{name}</b><small>{esc(model)}</small>'
                    f'<span class="mt-st-state">{STATE_TEXT[state]}</span></div></div>')
    return "".join(rows)


def friendly_error(msg: str):
    m = msg.lower()
    if "ffmpeg" in m:
        return "ffmpeg is missing", "Install ffmpeg (see the README) and restart the app."
    if "unsupported file type" in m:
        return "This file type isn't supported", f"Please upload one of: {FORMATS}."
    if "file is empty" in m:
        return "The file is empty", "The upload contains 0 bytes. Check the file and upload it again."
    if "could not be decoded" in m:
        return ("This file couldn't be read",
                "It may be corrupt, incomplete or have no audio track. Try exporting it again as MP3 or WAV.")
    if "silent" in m or "no speech" in m or "shorter than 1 second" in m:
        return "No speech found", "Make sure the recording contains spoken audio and isn't muted."
    if "too long" in m:
        return "Recording too long for this API key", "Use a Groq key with higher limits or a shorter recording."
    if "rate" in m and "limit" in m:
        return "Rate limit reached", "Groq's usage limit was hit. Wait a minute and try again."
    if "could not reach" in m:
        return "Connection problem", "Check your internet connection and try again."
    if "api key" in m or "groq_api_key" in m:
        return "API key problem", "Set GROQ_API_KEY as described in the README, then restart the app."
    return "Processing failed", "See the error message below. If it keeps happening, try another file."


def show_error(err: dict) -> None:
    st.markdown(f'<div class="mt-error"><div class="mt-error-title">⚠ {esc(err["title"])}</div>'
                f'<div class="mt-error-hint">Failed during {esc(err["stage"])}. {esc(err["hint"])}</div>'
                f'<div class="mt-error-msg">{esc(err["message"] or "(no message)")}</div></div>',
                unsafe_allow_html=True)
    if err.get("trace"):
        with st.expander("Technical details"):
            st.code(err["trace"], language=None)


def _paras(tokens) -> str:
    """Group (html, word) tokens into paragraphs of four sentences for easier reading."""
    out, cur, sentences = [], [], 0
    for html_tok, word in tokens:
        cur.append(html_tok)
        if word.endswith((".", "?", "!")):
            sentences += 1
            if sentences % 4 == 0:
                out.append("<p>" + " ".join(cur) + "</p>")
                cur = []
    if cur:
        out.append("<p>" + " ".join(cur) + "</p>")
    return "".join(out)


def paragraphs(text: str) -> str:
    return _paras([(esc(w), w) for w in text.split()])


@st.cache_data(show_spinner=False)
def diff_panels(raw: str, refined: str):
    """Word-level highlight of what the refiner changed (display only)."""
    a, b = raw.split(), refined.split()
    left, right = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            left += [(esc(w), w) for w in a[i1:i2]]
            right += [(esc(w), w) for w in b[j1:j2]]
        else:
            left += [(f'<span class="mt-del">{esc(w)}</span>', w) for w in a[i1:i2]]
            right += [(f'<span class="mt-ins">{esc(w)}</span>', w) for w in b[j1:j2]]
    return _paras(left), _paras(right)


def reset_app() -> None:
    for k in ("source", "raw", "refined", "changes", "blocked", "rec", "ui_steps", "ui_error"):
        ss.pop(k, None)
    ss.ui_uploader += 1


# ---------------- sidebar: upload, pipeline status, downloads ----------------
with st.sidebar:
    st.markdown('<div class="mt-side-title">1 · Recording</div>', unsafe_allow_html=True)
    file = st.file_uploader("Meeting recording (English audio or video)", type=None,
                            key=f"uploader_{ss.ui_uploader}", help=f"Accepted formats: {FORMATS}")
    st.caption(f"Accepted: {FORMATS}")
    if file is not None:
        ext = os.path.splitext(file.name)[1].lower()
        st.markdown(f'<div class="mt-fileinfo">📄 <b>{esc(file.name)}</b> <span>· {fmt_size(file.size)}</span></div>',
                    unsafe_allow_html=True)
        if ext not in ALLOWED:
            st.markdown(f'<div class="mt-warn">⚠ “{esc(ext or "no extension")}” is not an accepted format. '
                        'Processing will stop with an error.</div>', unsafe_allow_html=True)
        elif not file.size:
            st.markdown('<div class="mt-warn">⚠ This file is empty (0 bytes). Processing will stop '
                        'with an error.</div>', unsafe_allow_html=True)
        else:
            st.audio(file.getvalue(), format=file.type or "audio/wav")
    start = st.button("Start Processing", type="primary", disabled=file is None)
    st.markdown('<div class="mt-side-title">2 · Pipeline</div>', unsafe_allow_html=True)
    tracker = st.empty()
    side_actions = st.container()

# ---------------- header ----------------
st.markdown('<div class="mt-title">Lexicon <em>AI</em></div>'
            '<div class="mt-sub">Your AI meeting assistant. Turn a recording into a clean transcript, '
            'minutes, key decisions and action items.</div><div class="mt-rule"></div>',
            unsafe_allow_html=True)

# ---------------- processing (pipeline calls unchanged) ----------------
if start and file is not None:
    for k in ("source", "raw", "refined", "changes", "blocked", "rec"):
        st.session_state.pop(k, None)
    ss.pop("ui_error", None)
    st.session_state.source = file.name
    states = ["pending", "pending", "pending"]

    def set_step(i: int, state: str) -> None:
        states[i] = state
        tracker.markdown(stepper_html(states), unsafe_allow_html=True)

    step, current = "transcription", 0
    try:
        set_step(0, "running")
        with st.spinner(f"Step 1 of 3 · Transcribing the audio with {STT_MODEL}…"):
            st.session_state.raw = transcribe(file.getvalue(), file.name)
        set_step(0, "done")

        step, current = "refinement", 1
        set_step(1, "running")
        with st.spinner(f"Step 2 of 3 · Refining technical terms with {REFINER_MODEL}…"):
            (st.session_state.refined, st.session_state.changes,
             st.session_state.blocked) = refine(st.session_state.raw)
        set_step(1, "done")

        step, current = "minutes generation", 2
        set_step(2, "running")
        with st.spinner(f"Step 3 of 3 · Writing minutes, decisions and action items with {MINUTES_MODEL}…"):
            st.session_state.rec = generate_record(st.session_state.refined)
        set_step(2, "done")
        st.toast("Lexicon AI finished all three stages.", icon="✅")
    except PipelineError as e:
        set_step(current, "failed")
        title, hint = friendly_error(str(e))
        ss.ui_error = {"stage": step, "title": title, "hint": hint, "message": str(e)}
    except Exception as e:
        set_step(current, "failed")
        ss.ui_error = {"stage": step, "title": "Unexpected error",
                       "hint": "Something the app didn't anticipate went wrong; details are below.",
                       "message": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
    ss.ui_steps = list(states)

tracker.markdown(stepper_html(ss.get("ui_steps", ["pending"] * 3)), unsafe_allow_html=True)
rec = ss.get("rec")

if ss.get("ui_error"):
    show_error(ss.ui_error)

# ---------------- empty state ----------------
if "raw" not in ss and not ss.get("ui_error"):
    if file is None:
        icon, title = "🎧", "Welcome to Lexicon AI"
        text = ("Upload an English meeting recording (audio or video) in the left panel, "
                "then press <b>Start Processing</b>.")
    else:
        icon, title = "▶️", f"Ready to process {esc(file.name)}"
        text = "Press <b>Start Processing</b> in the left panel. Results will appear here."
    how = "".join(f'<div class="{tone(i * 2)}"><div class="mt-how-n">0{i + 1}</div><b>{name}</b>'
                  f'<p><code>{esc(model)}</code><br>{desc}</p></div>'
                  for i, (name, model, desc) in enumerate(STEPS))
    st.markdown(f'<div class="mt-empty"><div class="mt-empty-icon">{icon}</div>'
                f'<div class="mt-empty-title">{title}</div><div class="mt-empty-text">{text}</div></div>'
                f'<div class="mt-how">{how}</div>', unsafe_allow_html=True)

# ---------------- results ----------------
if "raw" in ss:
    st.markdown(f'<div class="mt-results"><span class="mt-results-title">Results</span>'
                f'<span class="mt-results-file">{esc(ss.source)}</span></div>', unsafe_allow_html=True)
    if file is not None and file.name != ss.source:
        note(f"ⓘ New file selected: <b>{esc(file.name)}</b>. Press Start Processing to analyse it; "
             "the results below are still for the previous file.")
    stats = [("Transcript words", len(ss.raw.split())),
             ("Term corrections", len(ss.changes) if "changes" in ss else "–"),
             ("Key decisions", len(rec.decisions) if rec is not None else "–"),
             ("Action items", len(rec.action_items) if rec is not None else "–")]
    st.markdown('<div class="mt-stats">' + "".join(
        f'<div class="{tone(i)}"><div class="mt-stat-value">{value}</div>'
        f'<div class="mt-stat-label">{label}</div></div>' for i, (label, value) in enumerate(stats))
        + "</div>", unsafe_allow_html=True)

    t_raw, t_ref, t_cmp, t_min, t_dec, t_act = st.tabs(
        ["Raw Transcript", "Refined Transcript", "Raw vs Refined", "Minutes & Summary",
         "Key Decisions", "Action Items"])
    not_ready = "Not available: processing stopped before this step (see the error above)."

    with t_raw:
        st.markdown(f'<div class="mt-label mt-c2">Speech-to-text output · {esc(STT_MODEL)}</div>'
                    f'<div class="mt-doc">{paragraphs(ss.raw)}</div>', unsafe_allow_html=True)

    with t_ref:
        if "refined" in ss:
            st.markdown(f'<div class="mt-label mt-c3">After terminology correction · {esc(REFINER_MODEL)}</div>'
                        f'<div class="mt-doc">{paragraphs(ss.refined)}</div>', unsafe_allow_html=True)
        else:
            note(not_ready)

    with t_cmp:
        if "refined" in ss:
            highlight = st.toggle("Highlight changed words", value=True)
            if highlight:
                left, right = diff_panels(ss.raw, ss.refined)
            else:
                left, right = paragraphs(ss.raw), paragraphs(ss.refined)
            c1, c2 = st.columns(2, gap="large")
            c1.markdown(f'<div class="mt-label mt-c1">Raw</div><div class="mt-doc">{left}</div>',
                        unsafe_allow_html=True)
            c2.markdown(f'<div class="mt-label mt-c3">Refined</div><div class="mt-doc">{right}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="mt-label mt-c0">Terminology corrections · {len(ss.changes)}</div>',
                        unsafe_allow_html=True)
            if ss.changes:
                st.markdown("".join(f'<div class="mt-change"><span class="mt-from">{esc(o)}</span>'
                                    f'<span class="mt-arrow">→</span><span class="mt-to">{esc(n)}</span></div>'
                                    for o, n in ss.changes), unsafe_allow_html=True)
            else:
                note("The refiner found no terminology errors to correct.")
            if ss.blocked:
                with st.expander(f"{len(ss.blocked)} edit(s) rejected by the safety check "
                                 "(would have changed numbers, negations, commitments, dates or content)"):
                    for x in ss.blocked:
                        st.text(x)
        else:
            note(not_ready)

    with t_min:
        if rec is None:
            note(not_ready)
        else:
            body = (f'<div class="mt-label mt-c0">Summary</div><div class="mt-lead">{esc(rec.summary)}</div>'
                    '<div class="mt-label mt-c1">Minutes</div>')
            for j, m in enumerate(rec.minutes):
                points = "".join(f"<li>{esc(p)}</li>" for p in m.points)
                body += f'<div class="mt-topic {tone(j)}">{esc(m.topic)}</div><ul class="mt-list">{points}</ul>'
            st.markdown(body, unsafe_allow_html=True)

    with t_dec:
        if rec is None:
            note(not_ready)
        elif rec.decisions:
            note("Only outcomes the participants explicitly agreed on. "
                 "Proposals that weren’t accepted appear in the minutes instead.")
            st.markdown("".join(f'<div class="mt-dec {tone(i - 1)}"><div class="mt-dec-n">{i:02d}</div>'
                                f'<div class="mt-dec-t">{esc(d)}</div></div>'
                                for i, d in enumerate(rec.decisions, 1)), unsafe_allow_html=True)
        else:
            note("No decisions were explicitly agreed in this recording.")

    with t_act:
        if rec is None:
            note(not_ready)
        elif rec.action_items:
            items = rec.action_items
            named = sum(not is_unspecified(a.owner) for a in items)
            dated = sum(not is_unspecified(a.deadline) for a in items)
            note(f"{len(items)} action items · {named} with a named owner · {dated} with a deadline. "
                 "“Unspecified” means the recording didn’t state it, so nothing is guessed.")
            rows = "".join(f'<tr><td class="mt-n {tone(i - 1)}">{i:02d}</td><td>{esc(a.task)}</td>'
                           f'<td>{who(a.owner)}</td><td>{when(a.deadline)}</td></tr>'
                           for i, a in enumerate(items, 1))
            st.markdown('<table class="mt-table"><thead><tr><th>#</th><th>Task</th><th>Owner</th>'
                        f'<th>Deadline</th></tr></thead><tbody>{rows}</tbody></table>',
                        unsafe_allow_html=True)
        else:
            note("No action items were assigned in this recording.")

# ---------------- sidebar actions (filled after processing so they appear immediately) ----------------
with side_actions:
    if "raw" in ss or ss.get("ui_error"):
        st.button("Process another file", on_click=reset_app, key="reset")
    if "raw" in ss:
        st.markdown('<div class="mt-side-title">3 · Downloads</div>', unsafe_allow_html=True)
        st.download_button("📝 Raw transcript (.txt)", ss.raw, "raw_transcript.txt", key="dl_raw")
        if "refined" in ss:
            st.download_button("✨ Refined transcript (.txt)", ss.refined, "refined_transcript.txt",
                               key="dl_ref")
        if rec is not None:
            st.download_button("📄 Meeting record (.md)", to_markdown(rec), "meeting_record.md",
                               mime="text/markdown", key="dl_md")
            st.download_button("🧩 Meeting record (.json)", to_json(rec), "meeting_record.json",
                               mime="application/json", key="dl_json")
            st.download_button("📦 All outputs (.zip)", to_zip(ss.raw, ss.refined, rec),
                               "meeting_outputs.zip", mime="application/zip", key="dl_zip")
import difflib, io, json, os, re, shutil, subprocess, tempfile, zipfile
from typing import List, Tuple

import groq
from groq import Groq
from pydantic import BaseModel
from pydub import AudioSegment
from prompts import REFINE_PROMPT, MINUTES_PROMPT, NOTES_PROMPT, LONG_INPUT_NOTE

# Models (check console.groq.com/docs/models if a name has changed)
STT_MODEL = "whisper-large-v3"
REFINER_MODEL = "openai/gpt-oss-20b"       # LLM #1: transcript refinement
MINUTES_MODEL = "openai/gpt-oss-120b"      # LLM #2: minutes, decisions, tasks
MODELS = {"speech_to_text": STT_MODEL, "refiner_llm": REFINER_MODEL, "minutes_llm": MINUTES_MODEL}

ALLOWED = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".opus", ".flac", ".webm", ".aac",
           ".wma", ".amr", ".mov", ".mkv", ".mpeg", ".mpga", ".3gp", ".aiff"}
CHUNK_MS = 10 * 60 * 1000


class PipelineError(Exception):
    pass


class TranscriptTooLong(PipelineError):
    pass


class Minute(BaseModel):
    topic: str
    points: List[str]


class ActionItem(BaseModel):
    task: str
    owner: str = "unspecified"
    deadline: str = "unspecified"


class MeetingRecord(BaseModel):
    summary: str
    minutes: List[Minute]
    decisions: List[str]
    action_items: List[ActionItem]


def client() -> Groq:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise PipelineError("GROQ_API_KEY is not set (see README > Setup).")
    return Groq(api_key=key, max_retries=6, timeout=120)  # SDK retries wait out 429 rate limits


def api_error(stage: str, e: Exception) -> PipelineError:
    if isinstance(e, groq.AuthenticationError):
        return PipelineError("The Groq API key was rejected - check GROQ_API_KEY.")
    if isinstance(e, groq.RateLimitError) or "Request too large" in str(e):
        return PipelineError(f"{stage}: this API key hit Groq's rate/token limit (free plan: 8K "
                             "tokens/min and 200K tokens/day per model). Wait a minute and retry, "
                             "or use a key with higher limits.")
    if isinstance(e, groq.APIConnectionError):
        return PipelineError(f"{stage}: could not reach the Groq API - check the internet connection.")
    return PipelineError(f"{stage} failed: {e}")


# ---------- Stage 1: speech to text ----------
def _decode(data: bytes, ext: str) -> AudioSegment:
    """ffmpeg converts any audio/video container to 16 kHz mono WAV, which pydub reads natively."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "input" + ext), os.path.join(d, "audio.wav")
        with open(src, "wb") as f:
            f.write(data)
        p = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst],
                           capture_output=True)
        if p.returncode != 0 or not os.path.exists(dst):
            raise ValueError(p.stderr.decode(errors="ignore"))
        return AudioSegment.from_wav(dst)


def load_audio(data: bytes, filename: str) -> AudioSegment:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED:
        raise PipelineError(f"Unsupported file type '{ext or 'no extension'}'. "
                            f"Please upload one of: {', '.join(sorted(ALLOWED))}")
    if not data:
        raise PipelineError("The uploaded file is empty (0 bytes).")
    if not shutil.which("ffmpeg"):
        raise PipelineError("ffmpeg is not installed on this machine, so audio cannot be decoded. "
                            "Install ffmpeg (see README) and restart the app.")
    try:
        audio = _decode(data, ext)
    except Exception:
        raise PipelineError("The file could not be decoded as audio: it may be corrupt, "
                            "truncated, or have no audio track.")
    if len(audio) < 1000:
        raise PipelineError("The recording is shorter than 1 second - nothing to transcribe.")
    if audio.max_dBFS < -50:  # -inf for digital silence
        raise PipelineError("The recording appears to be silent - no speech to transcribe.")
    return audio


def chunk_bounds(audio: AudioSegment) -> List[Tuple[int, int]]:
    """~10-minute chunks, each cut at the quietest point of its last 30 s so no word is split."""
    bounds, start = [], 0
    while len(audio) - start > CHUNK_MS:
        window = range(start + CHUNK_MS - 30_000, start + CHUNK_MS, 250)
        cut = min(window, key=lambda t: audio[t:t + 250].dBFS) + 125
        bounds.append((start, cut))
        start = cut
    bounds.append((start, len(audio)))
    return bounds


def _speech_text(r) -> str:
    """Drop segments Whisper itself flags as non-speech (stops 'Thank you.' hallucinations)."""
    segs = r.model_dump().get("segments") or []
    if not segs:
        return (r.text or "").strip()
    keep = [s.get("text", "").strip() for s in segs
            if not ((s.get("no_speech_prob") or 0) > 0.6 and (s.get("avg_logprob") or 0) < -1.0)]
    return " ".join(t for t in keep if t)


def transcribe(data: bytes, filename: str) -> str:
    audio = load_audio(data, filename)
    c = client()
    parts = []
    for start, end in chunk_bounds(audio):
        buf = io.BytesIO()
        audio[start:end].export(buf, format="mp3", bitrate="64k")
        try:
            r = c.audio.transcriptions.create(
                file=("chunk.mp3", buf.getvalue()), model=STT_MODEL, language="en",
                response_format="verbose_json", temperature=0.0)
        except Exception as e:
            raise api_error("Transcription", e)
        parts.append(_speech_text(r))
    raw = " ".join(p for p in parts if p).strip()
    if not raw:
        raise PipelineError("No speech was detected in the recording.")
    return raw


# ---------- Stage 2: refinement (LLM #1) ----------
def split_text(text: str, limit: int = 6000) -> List[str]:
    """Split on sentence boundaries into chunks of at most ~limit characters."""
    pieces = []
    for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        while len(s) > limit:  # long stretch with no sentence punctuation
            cut = s.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append(s[:cut])
            s = s[cut:].strip()
        if s:
            pieces.append(s)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > limit:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


_NUM = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
_NUM.update(thirty="30", forty="40", fifty="50", sixty="60", seventy="70", eighty="80",
            ninety="90", hundred="100", thousand="1000", million="1000000", billion="1000000000")
_KEEP = set("""not no never none nobody nothing neither nor without will shall should must
agree agreed approve approved monday tuesday wednesday thursday friday saturday sunday january
february march april may june july august september october november december today tomorrow
tonight yesterday""".split())


def _signature(words: List[str]):
    """Numbers, negations, commitments and dates in a span: refinement must not change these."""
    t = " ".join(words).lower().replace("\u2019", "'")
    t = t.replace("won't", "will not").replace("can't", "can not").replace("cannot", "can not")
    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"'ll\b", " will", t)
    toks = re.findall(r"[a-z]+|\d+", t)
    return (sorted(_NUM.get(x, x) for x in toks if x.isdigit() or x in _NUM),
            sorted(x for x in toks if x in _KEEP))


def _norm(words: List[str]) -> str:
    return re.sub(r"[^a-z0-9]", "", " ".join(words).lower())


def guard_refinement(raw: str, refined: str):
    """Keep the LLM's terminology fixes, undo any edit that adds/drops content or changes
    numbers, negations, commitments or dates. Returns (text, corrections, blocked_edits)."""
    a, b = raw.split(), refined.split()
    if not b or abs(len(a) - len(b)) > max(15, 0.15 * len(a)):
        return raw, [], ["A chunk was kept as raw text: the refiner changed its length too much "
                         "(possible summarising or truncation)."]
    out, changes, blocked = [], [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        old, new = a[i1:i2], b[j1:j2]
        if op == "equal":
            out += old
        elif len(new) - len(old) > 2 or len(old) - len(new) > 5 or _signature(old) != _signature(new):
            out += old
            blocked.append(f"{' '.join(old) or '(nothing)'}  ->  {' '.join(new) or '(deleted)'}")
        else:
            out += new
            if _norm(old) != _norm(new):  # don't log pure case/punctuation edits
                changes.append((" ".join(old), " ".join(new)))
    return " ".join(out), changes, blocked


def refine(raw: str):
    """LLM #1. Returns (refined_text, corrections, blocked_edits)."""
    c = client()
    out, changes, blocked = [], [], []
    for chunk in split_text(raw):
        try:
            r = c.chat.completions.create(
                model=REFINER_MODEL, temperature=0.0,
                messages=[{"role": "system", "content": REFINE_PROMPT},
                          {"role": "user", "content": chunk}])
        except Exception as e:
            raise api_error("Refinement", e)
        text, ch, bl = guard_refinement(chunk, (r.choices[0].message.content or "").strip())
        out.append(text)
        changes += ch
        blocked += bl
    return " ".join(out), changes, blocked


# ---------- Stage 3: minutes, decisions, tasks (LLM #2) ----------
_S = {"type": "string"}
RECORD_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["decisions", "action_items", "minutes", "summary"],
    "properties": {
        "decisions": {"type": "array", "items": _S},
        "action_items": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["task", "owner", "deadline"],
            "properties": {"task": _S, "owner": _S, "deadline": _S}}},
        "minutes": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["topic", "points"],
            "properties": {"topic": _S, "points": {"type": "array", "items": _S}}}},
        "summary": _S,
    },
}

_EMPTY = {"", "none", "n/a", "na", "null", "unknown", "tbd", "tba", "not specified",
          "unspecified", "not mentioned", "not stated", "unassigned", "someone", "anyone", "-"}
_FILLER = {"by", "on", "before", "until", "till", "the", "of", "end", "this", "next", "at",
           "in", "and", "to", "a", "an", "within", "team"}


def clean_value(v: str) -> str:
    v = (v or "").strip()
    return "unspecified" if v.lower() in _EMPTY else v


def grounded(value: str, transcript: str) -> str:
    """Keep an owner/deadline only if its key words actually occur in the transcript."""
    v = clean_value(value)
    if v == "unspecified":
        return v
    plain = lambda s: re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s.lower())
    t = plain(transcript)
    words = [w for w in re.findall(r"[a-z0-9]+", plain(v)) if w not in _FILLER]
    ok = bool(words) and all(re.search(rf"\b{re.escape(w)}", t) for w in words)
    return v if ok else "unspecified"


def _record(c: Groq, system: str, user: str) -> MeetingRecord:
    last_err = None
    for _ in range(3):
        try:
            r = c.chat.completions.create(
                model=MINUTES_MODEL, temperature=0.1,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "meeting_record", "strict": True, "schema": RECORD_SCHEMA}},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return MeetingRecord.model_validate_json(r.choices[0].message.content or "")
        except groq.APIStatusError as e:
            if e.status_code == 413 or "Request too large" in str(e):
                raise TranscriptTooLong("The meeting is too long for this API key's per-minute "
                                        "token limit, even after condensing. Use a Groq key with "
                                        "higher limits (Developer plan).") from e
            if isinstance(e, (groq.AuthenticationError, groq.RateLimitError)):
                raise api_error("Minutes generation", e) from e
            last_err = e  # other API errors: retry
        except groq.APIConnectionError as e:
            raise api_error("Minutes generation", e) from e
        except Exception as e:  # empty or invalid output: retry
            last_err = e
    raise PipelineError(f"Minutes generation failed after 3 attempts: {last_err}")


def _condense(c: Groq, refined: str) -> str:
    """Long meetings on low-limit keys: faithful notes per part (same LLM #2), then one record."""
    parts = split_text(refined, limit=10000)
    notes = []
    for i, part in enumerate(parts):
        context = f"[CONTEXT]\n{parts[i - 1][-1500:]}\n\n" if i else ""
        try:
            r = c.chat.completions.create(
                model=MINUTES_MODEL, temperature=0.1,
                messages=[{"role": "system", "content": NOTES_PROMPT},
                          {"role": "user", "content": f"{context}[PART {i + 1} of {len(parts)}]\n{part}"}])
        except Exception as e:
            raise api_error("Minutes generation (long meeting)", e)
        notes.append(f"Part {i + 1} notes:\n{(r.choices[0].message.content or '').strip()}")
    return "\n\n".join(notes)


def generate_record(refined: str) -> MeetingRecord:
    c = client()
    try:
        rec = _record(c, MINUTES_PROMPT, refined)
    except TranscriptTooLong:  # over the per-minute token cap: condense part by part, then retry
        rec = _record(c, MINUTES_PROMPT + LONG_INPUT_NOTE, _condense(c, refined))
    for a in rec.action_items:  # never report an owner/deadline the transcript doesn't contain
        a.owner = grounded(a.owner, refined)
        a.deadline = grounded(a.deadline, refined)
    return rec


# ---------- Rendering (both formats from the SAME object) ----------
def _cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def to_markdown(rec: MeetingRecord) -> str:
    md = ["# Meeting Record", "",
          f"_Models: {STT_MODEL} (speech-to-text) -> {REFINER_MODEL} (refinement) -> "
          f"{MINUTES_MODEL} (minutes)_", "", "## Summary", rec.summary, "", "## Minutes"]
    for m in rec.minutes:
        md.append(f"### {m.topic}")
        md += [f"- {p}" for p in m.points]
        md.append("")
    md.append("## Key Decisions")
    md += [f"{i}. {d}" for i, d in enumerate(rec.decisions, 1)] or ["None recorded."]
    md += ["", "## Action Items", "| # | Task | Owner | Deadline |", "|---|------|-------|----------|"]
    for i, a in enumerate(rec.action_items, 1):
        md.append(f"| {i} | {_cell(a.task)} | {_cell(a.owner)} | {_cell(a.deadline)} |")
    if not rec.action_items:
        md.append("| - | None recorded. | - | - |")
    return "\n".join(md)


def to_json(rec: MeetingRecord) -> str:
    return json.dumps({"models": MODELS, **rec.model_dump()}, indent=2, ensure_ascii=False)


def to_zip(raw: str, refined: str, rec: MeetingRecord) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("raw_transcript.txt", raw)
        z.writestr("refined_transcript.txt", refined)
        z.writestr("meeting_record.md", to_markdown(rec))
        z.writestr("meeting_record.json", to_json(rec))
    return buf.getvalue()
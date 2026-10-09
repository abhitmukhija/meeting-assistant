# LEXICON AI Meeting Assistant

Inter IIT Tech Meet 15.0 | Bootcamp Phase 2 | ML Problem Statement

An AI-powered meeting assistant that turns a recorded English meeting into an accurate transcript and a structured written record: summary, minutes, key decisions, and action items. A three-stage pipeline of one speech-to-text model and two separate language models does the work, behind an interactive Streamlit interface.

## Live App

https://abhitmukhija-meeting-assistant-app-4vpbp5.streamlit.app/

**No API key or installation is needed to use the live app.** Open the link, upload an English meeting recording, and click **Process recording**. An API key is only required if you want to [run the project locally](#local-setup-and-run).

Demo video: https://drive.google.com/file/d/13zr1fLOrOkxSXw0YBLr1XSTReDWVHVgJ/view?usp=drive_link

> Note: the live app runs on a free-tier Groq key with per-minute and per-day token limits. If you see a rate-limit message, wait a minute and retry. Very long recordings may need a key with higher limits.

---

## Features

- Upload an English meeting recording (audio or video) and process it in one run.
- Raw transcript, refined transcript, and final record are all kept and shown side by side.
- Domain-aware refinement fixes misheard technical terms and acronyms. A built-in safety check rejects any edit that changes numbers, negations, commitments, or dates.
- Structured minutes, key decisions, and action items. If an owner or deadline is not stated in the recording, it is shown as `unspecified` and never guessed.
- Clear error messages for unsupported, empty, corrupt, silent, or too-short files, and for API failures (bad key, rate limits, no connection).
- Downloads: raw transcript, refined transcript, meeting record in Markdown (human-readable) and JSON (machine-readable), and a ZIP of everything. Both record formats are generated from the same object, so they always contain identical decisions and tasks.
- Long meetings are handled by splitting audio into ~10 minute chunks (cut at the quietest point so no word is split) and by condensing very long transcripts before minutes generation.

---

## Models and Pipeline

| Stage | Role | Model | Provider |
|-------|------|-------|----------|
| 1 | Speech-to-text | `whisper-large-v3` | Groq |
| 2 | LLM #1: transcript refinement | `openai/gpt-oss-20b` | Groq |
| 3 | LLM #2: minutes, decisions, action items | `openai/gpt-oss-120b` | Groq |

The two language-model roles are separate stages with separate prompts (see `prompts.py`).

```
Audio/Video upload
      |
      v
[ffmpeg decode -> 16 kHz mono WAV -> ~10 min chunks]
      |
      v
Stage 1: Whisper large-v3 --------> RAW TRANSCRIPT
      |
      v
Stage 2: LLM #1 (gpt-oss-20b) ----> REFINED TRANSCRIPT
      |      (+ guard: rejects edits that change numbers, negations,
      |        commitments, dates, or add/drop content)
      v
Stage 3: LLM #2 (gpt-oss-120b) ---> SUMMARY, MINUTES, DECISIONS, ACTION ITEMS
      |      (strict JSON schema; owner/deadline kept only if they
      |        actually appear in the transcript, else "unspecified")
      v
Streamlit UI: view, compare, download (.txt, .md, .json, .zip)
```

How outputs move between stages:

1. `transcribe()` returns the raw transcript string. Segments that Whisper flags as non-speech are dropped to avoid hallucinated text on silence.
2. `refine()` receives the raw transcript, sends it to LLM #1 in sentence-aligned chunks, and returns the refined text plus the list of corrections and any blocked edits.
3. `generate_record()` receives only the refined transcript, sends it to LLM #2 with a strict JSON schema, and returns a validated `MeetingRecord` object. That one object is rendered to both Markdown and JSON.

---

## Project Structure

```
meeting-assistant/
├── app.py             # Streamlit interface (upload, status, results, downloads)
├── pipeline.py        # Transcription, refinement, record generation, rendering
├── prompts.py         # Prompts for the two LLM stages
├── requirements.txt   # Python dependencies
├── packages.txt       # System packages for Streamlit Community Cloud (ffmpeg)
├── .streamlit/        # Streamlit config; secrets.toml.example shows the key format
├── samples/           # Shareable sample recording and its generated outputs
└── .gitignore
```

---

## Prerequisites (for running locally)

The deployed app above needs none of this. These are only for running the project on your own machine.

- **Python 3.10 or newer**
- **ffmpeg** installed and available on your PATH (used to decode audio and video files)
- A **Groq API key** (free at https://console.groq.com/keys)

### Install ffmpeg

| OS | Command |
|----|---------|
| Windows | `winget install Gyan.FFmpeg` (then close and reopen the terminal) |
| macOS | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt update && sudo apt install ffmpeg` |

Check it works: `ffmpeg -version`

---

## Local Setup and Run

### 1. Clone the repository

```
git clone https://github.com/abhitmukhija/meeting-assistant.git
cd meeting-assistant
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, then activate again.

**Windows (Command Prompt):**

```
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Set your Groq API key

Use **one** of the options below. Never commit your key to git (`.env` and secrets files are in `.gitignore`).

**Option A: environment variable (current terminal session only)**

Windows PowerShell:
```
$env:GROQ_API_KEY = "your_key_here"
```

Windows Command Prompt:
```
set GROQ_API_KEY=your_key_here
```

macOS / Linux:
```
export GROQ_API_KEY="your_key_here"
```

**Option B: Streamlit secrets file**

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and put your key in it:

```toml
GROQ_API_KEY = "your_key_here"
```

`secrets.toml` is git-ignored, so your key is never committed.

### 5. Run the app

```
streamlit run app.py
```

The app opens at http://localhost:8501.

### 6. Use it

1. Upload an English meeting recording (see supported formats below).
2. Click **Process recording**.
3. Watch the three stages complete: transcription, refinement, minutes generation.
4. Open the tabs: **Transcripts** (raw vs refined side by side, plus the list of corrections), **Minutes**, **Key Decisions**, **Action Items**, **Downloads**.
5. Download the outputs you need.

---

## Supported File Formats

`.mp3 .wav .m4a .mp4 .ogg .opus .flac .webm .aac .wma .amr .mov .mkv .mpeg .mpga .3gp .aiff`

Unsupported, empty (0 bytes), corrupt, silent, or shorter-than-1-second files are rejected with a clear message in the interface.

---

## Outputs

For every processed recording the app displays and lets you download:

| Output | Format | Contents |
|--------|--------|----------|
| Raw transcript | `.txt` | Speech-to-text output as produced by Whisper |
| Refined transcript | `.txt` | Transcript after domain-aware terminology correction |
| Meeting record (human-readable) | `.md` | Summary, minutes, key decisions, action items table |
| Meeting record (machine-readable) | `.json` | Same content as the Markdown, plus the model names used |
| All outputs | `.zip` | All four files above |

JSON structure:

```json
{
  "models": { "speech_to_text": "...", "refiner_llm": "...", "minutes_llm": "..." },
  "summary": "...",
  "minutes": [ { "topic": "...", "points": ["..."] } ],
  "decisions": ["..."],
  "action_items": [ { "task": "...", "owner": "unspecified", "deadline": "unspecified" } ]
}
```

---

## How the Requirements Are Met

| Requirement | How it is handled |
|-------------|-------------------|
| Upload to final output in one run | A single **Process recording** click runs all three stages in order |
| Keep raw and refined transcripts | Both stored in session state, shown side by side, both downloadable |
| Clear errors for bad files | `PipelineError` messages shown in the UI for wrong type, empty, undecodable, silent, or too-short files |
| Two distinct LLM stages | Separate models and prompts: `gpt-oss-20b` (refine) then `gpt-oss-120b` (minutes) |
| Preserve names, numbers, negation, commitments | Refinement prompt plus a programmatic guard that reverts any risky edit |
| No invented owners or deadlines | Schema defaults to `unspecified`, and a grounding check keeps an owner or deadline only if its words appear in the transcript |
| No proposals presented as decisions | Enforced in the minutes prompt (`prompts.py`) |
| No hardcoded outputs | Every result comes from live model calls |
| Human-readable and machine-readable records | Markdown and JSON generated from the same object |
| Processing and failure status | `st.status` shows the current stage and which stage failed |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GROQ_API_KEY is not set` | Set the key (step 4) in the same terminal where you run `streamlit run` |
| `The Groq API key was rejected` | The key is wrong or revoked. Create a new one at console.groq.com |
| `ffmpeg is not installed` | Install ffmpeg, then close and reopen the terminal and restart the app |
| Rate or token limit message (live app or local) | Groq's free plan has per-minute and per-day token limits shared by everyone using the key. Wait a minute and retry, or run locally with your own key |
| Live app is asleep or slow to load | Streamlit Community Cloud puts idle apps to sleep. Click **Wake up** and wait about a minute |
| `streamlit` is not recognized | Activate the virtual environment, then run `pip install -r requirements.txt` again |
| Python 3.13 import error about `audioop` | Covered by `audioop-lts` in `requirements.txt`. Re-run the pip install |
| Very long meeting fails | Use a Groq key with higher limits. The app already condenses long transcripts first |

---

## Sample Run

Sample recording and the outputs generated by the app are in the `samples/` folder:

- `samples/meeting_audio.<ext>`
- `samples/raw_transcript.txt`
- `samples/refined_transcript.txt`
- `samples/meeting_record.md`
- `samples/meeting_record.json`

Meeting Recording drive folder link: https://drive.google.com/drive/folders/1rI3ubBfWvCHZSP1xqkhJ1twnLHyRNIZE?usp=drive_link
 
---

## Deployment (Streamlit Community Cloud)

The live app is hosted on Streamlit Community Cloud. To deploy your own copy:

1. Push this repository to GitHub (make sure `.env` and `secrets.toml` are **not** committed).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose this repository, branch `main`, and main file `app.py`.
4. Open **Advanced settings > Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
5. Click **Deploy**.

Python packages come from `requirements.txt`, and system packages (ffmpeg) are installed automatically from `packages.txt`. The API key is stored in Streamlit's secrets and is never visible to users of the app or in the repository.

---

## Team

| # | Name | Roll No. |
|---|------|----------|
| 1 | Aanvi Agarwal | 250122001 |
| 2 | Rimzhim Kudiwal | 250121045 |
| 3 | Abhit Mukhija | 250122002 |

IIT Guwahati Tech Board | Inter IIT Tech Meet 15.0

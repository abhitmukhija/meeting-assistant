import os
import streamlit as st
from pipeline import (transcribe, refine, generate_record, to_markdown, to_json, to_zip,
                      PipelineError, ALLOWED, STT_MODEL, REFINER_MODEL, MINUTES_MODEL)

st.set_page_config(page_title="AI Meeting Assistant", layout="wide")

try:  # also accept the key from .streamlit/secrets.toml (local) or Streamlit Cloud secrets
    os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    pass

st.title("AI Meeting Assistant")
st.caption(f"Speech-to-text: {STT_MODEL}  →  LLM 1 (terminology refinement): {REFINER_MODEL}  "
           f"→  LLM 2 (minutes, decisions, action items): {MINUTES_MODEL}")

file = st.file_uploader("Upload an English meeting recording", type=None,
                        help="Audio or video: " + ", ".join(sorted(ALLOWED)))
if file is not None and file.size and os.path.splitext(file.name)[1].lower() in ALLOWED:
    st.audio(file.getvalue(), format=file.type or "audio/wav")

if st.button("Process recording", type="primary"):
    if file is None:
        st.error("Please upload a file first.")
    else:
        for k in ("source", "raw", "refined", "changes", "blocked", "rec"):
            st.session_state.pop(k, None)
        st.session_state.source = file.name
        step = "transcription"
        try:
            with st.status("Processing...", expanded=True) as status:
                st.write(f"1/3 Transcribing audio ({STT_MODEL})...")
                st.session_state.raw = transcribe(file.getvalue(), file.name)
                step = "refinement"
                st.write(f"2/3 Refining domain-specific terms ({REFINER_MODEL})...")
                (st.session_state.refined, st.session_state.changes,
                 st.session_state.blocked) = refine(st.session_state.raw)
                step = "minutes generation"
                st.write(f"3/3 Generating minutes, decisions and tasks ({MINUTES_MODEL})...")
                st.session_state.rec = generate_record(st.session_state.refined)
                status.update(label="Done: all three stages completed", state="complete",
                              expanded=False)
        except PipelineError as e:
            status.update(label=f"Failed during {step}", state="error")
            st.error(str(e))
        except Exception as e:
            status.update(label=f"Failed during {step}", state="error")
            st.error(f"Unexpected error during {step}: {e}")

ss = st.session_state
if "raw" in ss:
    rec = ss.get("rec")
    st.subheader(f"Results for {ss.source}")
    t1, t2, t3, t4, t5 = st.tabs(["Transcripts", "Minutes", "Key Decisions", "Action Items", "Downloads"])

    with t1:
        a, b = st.columns(2)
        a.markdown("**Raw transcript** (speech-to-text output)")
        a.text_area("raw", ss.raw, height=400, label_visibility="collapsed")
        if "refined" in ss:
            b.markdown("**Refined transcript** (after terminology correction)")
            b.text_area("ref", ss.refined, height=400, label_visibility="collapsed")
            st.markdown(f"**Terminology corrections made by LLM 1: {len(ss.changes)}**")
            if ss.changes:
                st.table([{"Raw transcript": o, "Refined transcript": n} for o, n in ss.changes])
            else:
                st.caption("No terminology errors needed correcting.")
            if ss.blocked:
                with st.expander(f"{len(ss.blocked)} edit(s) rejected by the safety check "
                                 "(would have changed numbers, negations, commitments, dates or content)"):
                    for x in ss.blocked:
                        st.text(x)

    if rec is None:
        for t in (t2, t3, t4):
            t.info("Not available: processing stopped before this stage (see the error above).")
    else:
        with t2:
            st.subheader("Summary")
            st.write(rec.summary)
            for m in rec.minutes:
                st.markdown(f"**{m.topic}**")
                for p in m.points:
                    st.markdown(f"- {p}")
        with t3:
            if rec.decisions:
                for i, d in enumerate(rec.decisions, 1):
                    st.markdown(f"{i}. {d}")
            else:
                st.info("No decisions were explicitly agreed in this recording.")
        with t4:
            if rec.action_items:
                st.table([{"Task": x.task, "Owner": x.owner, "Deadline": x.deadline}
                          for x in rec.action_items])
            else:
                st.info("No action items were assigned in this recording.")

    with t5:
        st.download_button("Raw transcript (.txt)", ss.raw, "raw_transcript.txt")
        if "refined" in ss:
            st.download_button("Refined transcript (.txt)", ss.refined, "refined_transcript.txt")
        if rec is not None:
            st.download_button("Meeting record (.md, human-readable)", to_markdown(rec),
                               "meeting_record.md", mime="text/markdown")
            st.download_button("Meeting record (.json, machine-readable)", to_json(rec),
                               "meeting_record.json", mime="application/json")
            st.download_button("All outputs (.zip)", to_zip(ss.raw, ss.refined, rec),
                               "meeting_outputs.zip", mime="application/zip")
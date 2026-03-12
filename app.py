import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from prompt_enhancer import enhance

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NegativePrompt Enhancer",
    page_icon="⚡",
    layout="centered",
)

# ── CSS — dark ChatGPT-like theme ─────────────────────────────────────────────
st.markdown("""
<style>
  /* Global background */
  .stApp { background-color: #212121; }

  /* Hide Streamlit chrome */
  header[data-testid="stHeader"],
  footer { display: none !important; }
  .block-container { padding-top: 0 !important; }

  /* ── Title bar ── */
  .title-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 28px 0 8px 0;
  }
  .title-bar h1 {
    color: #ececec;
    font-size: 1.55rem;
    font-weight: 600;
    margin: 0;
    font-family: "Söhne", ui-sans-serif, system-ui, sans-serif;
  }

  /* ── Chat bubbles ── */
  .chat-row {
    display: flex;
    margin: 18px 0;
    gap: 14px;
    max-width: 720px;
    margin-left: auto;
    margin-right: auto;
  }
  .chat-row.user  { flex-direction: row-reverse; }
  .chat-row.bot   { flex-direction: row; }

  .avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
    margin-top: 2px;
  }
  .avatar.user { background: #19c37d; color: #fff; }
  .avatar.bot  { background: #444654; color: #fff; }

  .bubble {
    padding: 12px 16px;
    border-radius: 14px;
    font-size: 0.95rem;
    line-height: 1.6;
    max-width: 84%;
    word-break: break-word;
  }
  .bubble.user {
    background: #2f2f2f;
    color: #ececec;
    border-bottom-right-radius: 4px;
  }
  .bubble.bot {
    background: #343541;
    color: #ececec;
    border-bottom-left-radius: 4px;
  }

  /* ── Task badge ── */
  .task-badge {
    display: inline-block;
    background: #19c37d22;
    color: #19c37d;
    border: 1px solid #19c37d55;
    border-radius: 20px;
    padding: 2px 11px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 8px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }

  /* ── Enhanced prompt block ── */
  .enhanced-block {
    background: #40414f;
    border-left: 3px solid #19c37d;
    border-radius: 8px;
    padding: 12px 14px;
    font-family: "Söhne Mono", ui-monospace, monospace;
    font-size: 0.9rem;
    color: #ececec;
    white-space: pre-wrap;
    word-break: break-word;
    margin-top: 4px;
  }

  /* ── Score row ── */
  .score-row {
    margin-top: 10px;
    font-size: 0.8rem;
    color: #8e8ea0;
  }
  .score-val { color: #ececec; font-weight: 600; }
  .score-gain { color: #19c37d; }

  /* ── Input area ── */
  .input-wrapper {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #212121;
    padding: 14px 0 20px 0;
    z-index: 100;
  }
  .input-inner {
    max-width: 720px;
    margin: 0 auto;
    padding: 0 16px;
  }

  /* Override Streamlit's text_area */
  textarea[data-testid="stChatInputTextArea"],
  .stChatInput textarea {
    background: #40414f !important;
    border: 1px solid #565869 !important;
    border-radius: 12px !important;
    color: #ececec !important;
    font-size: 0.95rem !important;
    resize: none !important;
  }
  .stChatInput > div { background: transparent !important; }

  /* Scrollable chat area */
  .chat-area {
    max-width: 720px;
    margin: 0 auto;
    padding: 0 16px 140px 16px;
  }

  /* Welcome screen */
  .welcome {
    text-align: center;
    color: #8e8ea0;
    font-size: 1rem;
    margin-top: 80px;
  }
  .welcome h2 {
    color: #ececec;
    font-size: 1.8rem;
    font-weight: 600;
    margin-bottom: 10px;
  }
  .welcome .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 24px;
  }
  .chip {
    background: #2f2f2f;
    border: 1px solid #3e3e3e;
    border-radius: 20px;
    padding: 7px 16px;
    font-size: 0.83rem;
    color: #ababab;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-bar">
  <span style="font-size:1.5rem">⚡</span>
  <h1>NegativePrompt Enhancer</h1>
</div>
""", unsafe_allow_html=True)

# ── Chat area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
      <h2>Améliore ton prompt</h2>
      <p>Écris n'importe quel texte — le modèle détecte la tâche<br>
         et applique le stimulus négatif optimal.</p>
      <div class="chips">
        <span class="chip">💬 Analyse de sentiment</span>
        <span class="chip">🔄 Antonymes</span>
        <span class="chip">🇫🇷 Traduction EN→FR</span>
        <span class="chip">⚡ Cause & Effet</span>
        <span class="chip">🐘 Plus grand animal</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-row user">
              <div class="avatar user">U</div>
              <div class="bubble user">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            r = msg["result"]
            gain = r.expected_score - r.baseline_score
            gain_str = f"+{gain:.2f}" if gain >= 0 else f"{gain:.2f}"
            gain_color = "#19c37d" if gain >= 0 else "#ff6b6b"
            st.markdown(f"""
            <div class="chat-row bot">
              <div class="avatar bot">N</div>
              <div class="bubble bot">
                <span class="task-badge">{r.task}</span><br>
                <div class="enhanced-block">{r.enhanced_prompt}</div>
                <div class="score-row">
                  Score attendu&nbsp;:&nbsp;<span class="score-val">{r.expected_score:.2f}</span>
                  &nbsp;·&nbsp; baseline&nbsp;:&nbsp;<span class="score-val">{r.baseline_score:.2f}</span>
                  &nbsp;·&nbsp; gain&nbsp;:&nbsp;<span style="color:{gain_color};font-weight:600">{gain_str}</span>
                  &nbsp;·&nbsp; stimulus&nbsp;NP{r.pnum}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
prompt = st.chat_input("Écris ton prompt ici…")

if prompt and prompt.strip():
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt.strip()})

    # Enhance
    try:
        result = enhance(prompt.strip())
        st.session_state.messages.append({"role": "bot", "result": result})
    except Exception as e:
        st.session_state.messages.append({
            "role": "bot",
            "result": type("R", (), {
                "task": "erreur",
                "enhanced_prompt": f"Erreur : {e}",
                "expected_score": 0, "baseline_score": 0, "pnum": 0,
            })()
        })

    st.rerun()

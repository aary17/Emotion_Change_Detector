"""
Emotion Change Detection in Conversation — Live Chat
A polished, single-file Streamlit application.

Run with:  streamlit run app.py
"""

import re
import time
import random

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
from conversation_analyzer import analyze_message

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Change Detection — Live Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------------------
# EMOTION MODEL (lightweight lexicon-based classifier — no external downloads needed)
# --------------------------------------------------------------------------------------
EMOTIONS = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Disgust", "Neutral"]

EMOTION_META = {
    "Joy":      {"icon": "😊", "color": "#34d399", "valence": 2.0},
    "Surprise": {"icon": "😲", "color": "#38bdf8", "valence": 1.0},
    "Neutral":  {"icon": "😐", "color": "#94a3b8", "valence": 0.0},
    "Sadness":  {"icon": "😞", "color": "#818cf8", "valence": -1.2},
    "Fear":     {"icon": "😨", "color": "#c084fc", "valence": -1.3},
    "Disgust":  {"icon": "🤢", "color": "#a3e635", "valence": -1.5},
    "Anger":    {"icon": "😡", "color": "#f87171", "valence": -2.0},
}

LEXICON = {
    "Joy": ["excited", "happy", "great", "love", "wonderful", "glad", "awesome",
            "amazing", "delighted", "thank you", "thanks", "fantastic", "perfect",
            "yay", "nice", "pleased", "grateful", "looking forward", "cool", "excellent"],
    "Sadness": ["sad", "unfortunately", "sorry", "disappointed", "cancelled", "canceled",
                "lost", "regret", "unhappy", "upset", "down", "miss", "heartbroken",
                "cry", "depressed", "hurt", "let down"],
    "Anger": ["ridiculous", "angry", "furious", "unacceptable", "outrageous", "annoyed",
              "frustrated", "mad", "hate", "sick of", "worst", "terrible service",
              "how dare", "absurd", "disgraceful", "livid", "rage"],
    "Fear": ["worried", "scared", "afraid", "nervous", "anxious", "concerned",
             "panic", "terrified", "uneasy", "dread"],
    "Surprise": ["wow", "really?", "no way", "unexpected", "shocked", "surprised",
                 "can't believe", "cannot believe", "unbelievable", "whoa"],
    "Disgust": ["gross", "disgusting", "awful", "nasty", "repulsive", "yuck"],
}

NEGATORS = {"not", "never", "no", "n't"}


def classify_emotion(text: str):
    """Very small, transparent lexicon-based emotion classifier with a confidence score."""
    lowered = text.lower()

    scores = {e: 0.0 for e in EMOTIONS}
    for emotion, keywords in LEXICON.items():
        for kw in keywords:
            if kw in lowered:
                weight = 1.0
                idx = lowered.find(kw)
                preceding = lowered[max(0, idx - 12):idx]
                if any(neg in preceding for neg in NEGATORS):
                    weight = -0.6
                scores[emotion] += weight

    if text.strip().endswith("!"):
        for e in ["Anger", "Joy", "Surprise"]:
            if scores[e] > 0:
                scores[e] += 0.3

    total_signal = sum(max(s, 0) for s in scores.values())

    if total_signal <= 0:
        scores["Neutral"] = 1.0
        total_signal = 1.0

    best_emotion = max(scores, key=lambda e: scores[e])
    if scores[best_emotion] <= 0:
        best_emotion = "Neutral"

    raw = max(scores[best_emotion], 0.35)
    confidence = min(0.99, 0.55 + (raw / (raw + 1.5)))
    confidence = round(confidence * 100)
    return best_emotion, confidence


# --------------------------------------------------------------------------------------
# LIGHTWEIGHT REPLY GENERATOR (rule-based — no ML/training involved)
# --------------------------------------------------------------------------------------
REPLIES = {
    "Joy": ["That's wonderful to hear! 😊", "Love the positive energy — tell me more!",
            "Great, I'm glad things are going well."],
    "Sadness": ["I'm sorry to hear that.", "That sounds really tough — I'm here.",
                "That's disappointing. Want to tell me more?"],
    "Anger": ["I understand your frustration.", "That does sound frustrating — let's sort it out.",
              "I hear you, and that's a fair thing to be upset about."],
    "Fear": ["It's okay to feel uneasy about that.", "Let's take this one step at a time.",
              "That sounds worrying — I'm listening."],
    "Surprise": ["Wow, that is surprising!", "Didn't see that coming — what happened next?"],
    "Disgust": ["That does sound unpleasant.", "I get why that would bother you."],
    "Neutral": ["Got it — tell me more.", "I see. What happened next?", "Understood, go on."],
}


def generate_reply(emotion: str) -> str:
    return random.choice(REPLIES[emotion])


# --------------------------------------------------------------------------------------
# STYLES
# --------------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, rgba(139,92,246,0.18), transparent 60%),
                radial-gradient(1000px 500px at 100% 0%, rgba(56,189,248,0.14), transparent 55%),
                linear-gradient(180deg, #0b0f1e 0%, #0a0e1a 100%);
}
#MainMenu, footer, header {visibility: hidden;}

@keyframes fadeInUp { from { opacity:0; transform: translateY(14px);} to {opacity:1; transform:translateY(0);} }
@keyframes fadeIn { from {opacity:0;} to {opacity:1;} }
@keyframes pulseGlow { 0%,100% {opacity:0.55;} 50% {opacity:1;} }
@keyframes popIn { 0% {opacity:0; transform:scale(0.85);} 60% {opacity:1; transform:scale(1.05);} 100% {transform:scale(1);} }
@keyframes shimmerFlash { 0% {box-shadow:0 0 0 rgba(139,92,246,0);} 30% {box-shadow:0 0 30px rgba(139,92,246,0.55);} 100% {box-shadow:0 0 0 rgba(139,92,246,0);} }
@keyframes bounce { 0%,80%,100% {transform: translateY(0);} 40% {transform: translateY(-6px);} }

.glass-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.055), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px 20px;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    animation: fadeInUp 0.5s ease both;
}
.glass-card:hover { border-color: rgba(139,92,246,0.35); transition: 0.25s ease; }

.hero-wrap { text-align:center; margin: 6px 0 10px 0; }
.hero-badge {
    display:inline-flex; align-items:center; gap:8px;
    background: rgba(139,92,246,0.12);
    border: 1px solid rgba(139,92,246,0.3);
    color:#c4b5fd; padding:6px 16px; border-radius:999px;
    font-size:12.5px; font-weight:600; letter-spacing:0.04em;
    animation: fadeIn 0.6s ease both;
}
.hero-dot { width:7px; height:7px; border-radius:50%; background: linear-gradient(90deg,#8b5cf6,#38bdf8); animation: pulseGlow 1.8s ease-in-out infinite; }
.hero-title {
    font-size:60px; font-weight:900; letter-spacing:-0.03em; line-height:1.05;
    background: linear-gradient(90deg,#ffffff 15%,#c4b5fd 55%,#7dd3fc 90%);
    -webkit-background-clip:text; background-clip:text; color:transparent;
    margin: 16px 0 8px 0; animation: fadeInUp 0.6s ease both;
}
.hero-sub { color:#94a3b8; font-size:17px; animation: fadeInUp 0.7s ease both; }

.section-title { font-size:20px; font-weight:700; color:#f1f5f9; margin: 4px 0 10px 0; display:flex; align-items:center; gap:8px; }

.stat-value { font-size:24px; font-weight:800; color:#f8fafc; }
.stat-label { font-size:12px; color:#93a0b8; text-transform:uppercase; letter-spacing:0.05em; }

.chip-row { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
.chip { background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.09); border-radius:999px;
        padding:4px 10px; font-size:13px; color:#cbd5e1; animation: popIn 0.35s ease both; }
.chip-arrow { color:#64748b; font-size:13px; }

.bubble-row { display:flex; margin: 10px 0; animation: fadeInUp 0.35s ease both; }
.bubble-row.user { justify-content:flex-end; }
.bubble-row.system { justify-content:flex-start; }
.bubble {
    max-width: 72%; padding: 13px 17px; border-radius: 18px; font-size:15.5px; line-height:1.5;
    box-shadow: 0 6px 18px rgba(0,0,0,0.22);
}
.bubble.user {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color:white; border-bottom-right-radius:6px;
}
.bubble.system {
    background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.09);
    color:#e2e8f0; border-bottom-left-radius:6px;
}
.bubble-meta { display:flex; gap:8px; align-items:center; margin-top:6px; font-size:12.5px; opacity:0.9; }
.bubble-emotion-badge {
    display:inline-flex; align-items:center; gap:5px; padding:3px 9px; border-radius:999px;
    font-weight:700; font-size:12px; border:1px solid rgba(255,255,255,0.18);
    animation: popIn 0.4s ease both;
}

.typing-wrap { display:flex; justify-content:flex-start; margin:10px 0; animation: fadeIn 0.3s ease both; }
.typing-bubble { background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.09);
    border-radius:18px; border-bottom-left-radius:6px; padding:14px 18px; display:flex; gap:5px; }
.typing-dot { width:7px; height:7px; border-radius:50%; background:#94a3b8; animation: bounce 1.2s infinite ease-in-out; }
.typing-dot:nth-child(2) { animation-delay: 0.15s; }
.typing-dot:nth-child(3) { animation-delay: 0.3s; }

.shift-banner {
    background: linear-gradient(135deg, rgba(139,92,246,0.18), rgba(56,189,248,0.1));
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 16px; padding: 14px 18px; margin: 10px 0 16px 0;
    animation: shimmerFlash 1.6s ease, fadeInUp 0.4s ease both;
    font-size:14.5px; color:#e9e3ff;
}

div.stButton > button {
    background: linear-gradient(90deg,#7c3aed,#38bdf8);
    color:white; border:none; border-radius:12px;
    padding: 0.55em 1.3em; font-weight:700; letter-spacing:0.01em;
    box-shadow: 0 6px 20px rgba(124,58,237,0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 26px rgba(124,58,237,0.5); }
div.stButton > button:active { transform: translateY(0px) scale(0.98); }

[data-testid="stChatInput"] textarea, div[data-baseweb="textarea"] textarea {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 14px !important;
    color:#e2e8f0 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
div[data-baseweb="textarea"] textarea:focus {
    border-color: rgba(139,92,246,0.5) !important;
    box-shadow: 0 0 12px rgba(139,92,246,0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []          # list of {role, text, emotion?, confidence?}
if "user_emotions" not in st.session_state:
    st.session_state.user_emotions = [] # list of {emotion, confidence, text}
if "pending_reply" not in st.session_state:
    st.session_state.pending_reply = False
if "last_shift" not in st.session_state:
    st.session_state.last_shift = None  # (prev_emotion, cur_emotion) shown once
if "pasted_analysis" not in st.session_state:
    st.session_state.pasted_analysis = None
if "pasted_input_text" not in st.session_state:
    st.session_state.pasted_input_text = ""


def emotion_delta(prev_e, cur_e):
    return abs(EMOTION_META[cur_e]["valence"] - EMOTION_META[prev_e]["valence"])


def handle_new_message(text: str):
    # Step 1: Detect emotion using the emotion detector
    result = analyze_message(
        text,
        st.session_state.user_emotions
    )

    emotion = result["emotion"]
    confidence = round(result["confidence"] * 100)

    # Step 2: Add user message to chat
    st.session_state.chat.append({
        "role": "user",
        "text": text,
        "emotion": emotion,
        "confidence": confidence
    })

    # Step 3: Get transition information
    transition = result["transition"]

    if transition["changed"] and transition["significant"]:
        st.session_state.last_shift = (
            transition["from"],
            transition["to"]
        )
    else:
        st.session_state.last_shift = None

    # Step 4: Store emotion for future comparisons
    st.session_state.user_emotions.append({
        "emotion": emotion,
        "confidence": confidence,
        "text": text
    })

    # Step 5: Trigger system reply
    st.session_state.pending_reply = True


def analyze_pasted_transcript(raw_text: str):
    if not raw_text or not raw_text.strip():
        return None

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    parsed_turns = []
    pasted_history = []

    for idx, line in enumerate(lines, 1):
        # Extract speaker prefix if present (e.g., "Customer:", "Agent:", "User:", "Speaker 1:")
        match = re.match(r'^(?:(User|Client|Customer|Agent|Support|Friend|Speaker\s*\d+|Person\s*\d+|[A-Z][a-z]+)\s*:\s*)?(.*)$', line, re.IGNORECASE)
        if match and match.group(1) and match.group(2):
            speaker = match.group(1).capitalize()
            cleaned_text = match.group(2).strip()
        else:
            speaker = f"Turn {idx}"
            cleaned_text = line.strip()

        if not cleaned_text:
            cleaned_text = line

        # Analyze message with isolated pasted_history
        result = analyze_message(cleaned_text, pasted_history)
        emotion = result["emotion"]
        confidence = round(result["confidence"] * 100)
        transition = result["transition"]

        turn_data = {
            "turn": idx,
            "speaker": speaker,
            "text": cleaned_text,
            "raw_line": line,
            "emotion": emotion,
            "confidence": confidence,
            "transition": transition,
            "valence": EMOTION_META[emotion]["valence"],
            "color": EMOTION_META[emotion]["color"],
            "icon": EMOTION_META[emotion]["icon"]
        }

        parsed_turns.append(turn_data)
        pasted_history.append({
            "emotion": emotion,
            "confidence": confidence,
            "text": cleaned_text
        })

    # Calculate Summary Metrics
    turns_count = len(parsed_turns)
    shifts_count = sum(1 for t in parsed_turns if t["transition"]["changed"])
    initial_emotion = parsed_turns[0]["emotion"] if turns_count else "Neutral"
    final_emotion = parsed_turns[-1]["emotion"] if turns_count else "Neutral"

    emotion_counts = {}
    for t in parsed_turns:
        emotion_counts[t["emotion"]] = emotion_counts.get(t["emotion"], 0) + 1
    dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "Neutral"

    return {
        "turns": parsed_turns,
        "turns_count": turns_count,
        "shifts_count": shifts_count,
        "initial_emotion": initial_emotion,
        "final_emotion": final_emotion,
        "dominant_emotion": dominant_emotion,
        "emotion_counts": emotion_counts
    }


# --------------------------------------------------------------------------------------
# HERO
# --------------------------------------------------------------------------------------
st.markdown("""
<div class="hero-wrap">
  <div class="hero-title">Emotion Change Detection</div>
  <div class="hero-sub">Analyze live conversations or track emotional trajectories across full transcript histories.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# --------------------------------------------------------------------------------------
tab_chat, tab_pasted = st.tabs([
    "💬 Live Chat Intelligence",
    "📜 Pasted Conversation Analysis"
])

# ======================================================================================
# TAB 1: LIVE CHAT INTELLIGENCE
# ======================================================================================
with tab_chat:
    turns = len(st.session_state.user_emotions)
    changes = sum(
        1 for i in range(1, turns)
        if st.session_state.user_emotions[i]["emotion"] != st.session_state.user_emotions[i - 1]["emotion"]
    )
    current_emotion = st.session_state.user_emotions[-1]["emotion"] if turns else "Neutral"
    cur_meta = EMOTION_META[current_emotion]

    d1, d2, d3, d4 = st.columns([1.3, 1, 1, 0.7])
    with d1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-label">Current Emotion</div>
            <div class="stat-value" style="color:{cur_meta['color']}">{cur_meta['icon']} {current_emotion if turns else '—'}</div>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-label">Emotion Changes</div>
            <div class="stat-value">{changes}</div>
        </div>""", unsafe_allow_html=True)
    with d3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-label">Conversation Turns</div>
            <div class="stat-value">{turns}</div>
        </div>""", unsafe_allow_html=True)
    with d4:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("↺ Reset", use_container_width=True, key="reset_live_chat"):
            st.session_state.chat = []
            st.session_state.user_emotions = []
            st.session_state.pending_reply = False
            st.session_state.last_shift = None
            st.rerun()

    # Recent emotion changes chip row
    if turns:
        recent = st.session_state.user_emotions[-6:]
        chips = []
        for i, e in enumerate(recent):
            meta = EMOTION_META[e["emotion"]]
            chips.append(f'<span class="chip" style="border-color:{meta["color"]}55">{meta["icon"]} {e["emotion"]}</span>')
            if i < len(recent) - 1:
                chips.append('<span class="chip-arrow">→</span>')
        st.markdown(f'<div class="chip-row" style="margin:10px 0 4px 0;">{"".join(chips)}</div>', unsafe_allow_html=True)

    # Significant shift banner
    if st.session_state.last_shift:
        prev_e, cur_e = st.session_state.last_shift
        pm, cm = EMOTION_META[prev_e], EMOTION_META[cur_e]
        st.markdown(f"""
        <div class="shift-banner">
            ⚡ <strong>Significant emotion shift detected:</strong>
            <span style="color:{pm['color']}">{pm['icon']} {prev_e}</span> → <span style="color:{cm['color']}">{cm['icon']} {cur_e}</span>
        </div>
        """, unsafe_allow_html=True)

    # Collapsible Emotion Journey
    with st.expander("📈 Live Emotion Journey", expanded=False):
        if turns < 2:
            st.markdown("<div style='color:#7c8aa5; font-size:13.5px;'>Send a couple of messages to see your emotion timeline.</div>", unsafe_allow_html=True)
        else:
            xs = list(range(1, turns + 1))
            vals = [EMOTION_META[e["emotion"]]["valence"] for e in st.session_state.user_emotions]
            cols = [EMOTION_META[e["emotion"]]["color"] for e in st.session_state.user_emotions]
            hover = [f"{e['emotion']}<br>{e['text']}" for e in st.session_state.user_emotions]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xs, y=vals, mode="lines+markers",
                line=dict(color="rgba(139,92,246,0.55)", width=2, shape="spline"),
                marker=dict(size=14, color=cols, line=dict(color="rgba(255,255,255,0.4)", width=1)),
                hoverinfo="text", hovertext=hover,
            ))
            fig.update_layout(
                height=240, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Turn", tickmode="linear", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Valence", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)"),
                font=dict(color="#e2e8f0"),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 Live Chat Window</div>', unsafe_allow_html=True)
    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat:
            st.markdown("""
            <div class="glass-card" style="text-align:center; color:#7c8aa5;">
                Start the live conversation below — send a message and watch the emotion detector work in real time.
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.chat:
            if msg["role"] == "user":
                meta = EMOTION_META[msg["emotion"]]
                st.markdown(f"""
                <div class="bubble-row user">
                    <div class="bubble user" style="border-right:3px solid {meta['color']}">
                        {msg['text']}
                        <div class="bubble-meta">
                            <span class="bubble-emotion-badge" style="background:{meta['color']}22; color:{meta['color']}">
                                {meta['icon']} {msg['emotion']}
                            </span>
                            <span style="color:rgba(255,255,255,0.75)">Confidence: {msg['confidence']}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bubble-row system">
                    <div class="bubble system">{msg['text']}</div>
                </div>
                """, unsafe_allow_html=True)

        if st.session_state.pending_reply:
            st.markdown("""
            <div class="typing-wrap">
                <div class="typing-bubble">
                    <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div id="chat-bottom-anchor"></div>', unsafe_allow_html=True)

    # Smooth-scroll to newest message
    components.html("""
    <script>
    try {
      const doc = window.parent.document;
      const anchor = doc.getElementById('chat-bottom-anchor');
      if (anchor) { anchor.scrollIntoView({behavior: 'smooth', block: 'end'}); }
    } catch (e) {}
    </script>
    """, height=0)

    user_input = st.chat_input("Live Chat: Type a message and press Enter...")
    if user_input and not st.session_state.pending_reply:
        handle_new_message(user_input)
        st.rerun()

    # Generate system reply after user message
    if st.session_state.pending_reply:
        time.sleep(0.7)
        last_emotion = st.session_state.chat[-1]["emotion"]
        reply = generate_reply(last_emotion)
        st.session_state.chat.append({"role": "system", "text": reply})
        st.session_state.pending_reply = False
        st.rerun()


# ======================================================================================
# TAB 2: PASTED CONVERSATION ANALYSIS (SEPARATE & ISOLATED)
# ======================================================================================
with tab_pasted:
    st.markdown('<div class="section-title">📜 Pasted Transcript Analysis</div>', unsafe_allow_html=True)
    st.markdown("<div style='color:#94a3b8; font-size:14px; margin-bottom:12px;'>Paste a full multi-line conversation to generate an independent analysis of emotional progression and transitions without affecting your live chat history.</div>", unsafe_allow_html=True)

    # Quick Sample Loaders
    col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
    with col_s1:
        if st.button("📋 Load Customer Service Sample", use_container_width=True):
            st.session_state.pasted_input_text = (
                "Customer: I've been waiting for 3 days and nobody responded to my urgent ticket!\n"
                "Customer: This service is absolutely unacceptable and infuriating.\n"
                "Agent: I am deeply sorry for the delay. Let me take over and fix this immediately.\n"
                "Customer: Okay, thank you. I appreciate you looking into this.\n"
                "Customer: Wow, it is resolved already? That is fantastic and wonderful news! 😊"
            )
            st.rerun()
    with col_s2:
        if st.button("📋 Load News Reaction Sample", use_container_width=True):
            st.session_state.pasted_input_text = (
                "User: Did you see the latest news about the sudden severe storm warning?\n"
                "Friend: Really? I had no idea, that sounds really frightening and concerning.\n"
                "User: Don't worry, the updated weather report says the storm changed direction.\n"
                "Friend: Oh what a huge relief! That's excellent news!"
            )
            st.rerun()

    pasted_text_input = st.text_area(
        "Conversation Transcript Input",
        value=st.session_state.get("pasted_input_text", ""),
        placeholder="Paste full conversation transcript here, e.g.:\nCustomer: Hi, I need help with my order.\nCustomer: I've been waiting for 3 days and nobody responded!\nAgent: I'm sorry to hear that, let me check.\nCustomer: Wow, thanks for fixing it so quickly!",
        height=160,
        key="pasted_text_area_widget"
    )

    b1, b2 = st.columns([2, 1])
    with b1:
        if st.button("⚡ Analyze Conversation Trajectory", use_container_width=True, key="analyze_pasted_btn"):
            if pasted_text_input and pasted_text_input.strip():
                st.session_state.pasted_analysis = analyze_pasted_transcript(pasted_text_input)
                st.session_state.pasted_input_text = pasted_text_input
                st.rerun()
            else:
                st.warning("Please paste a conversation transcript before analyzing.")
    with b2:
        if st.button("🗑️ Clear Analysis", use_container_width=True, key="clear_pasted_btn"):
            st.session_state.pasted_analysis = None
            st.session_state.pasted_input_text = ""
            st.rerun()

    # RENDER ANALYSIS RESULTS
    if st.session_state.get("pasted_analysis"):
        analysis = st.session_state.pasted_analysis
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Conversation Analysis & Emotion Arc</div>', unsafe_allow_html=True)

        # 1. Summary Metrics Cards
        m1, m2, m3, m4 = st.columns(4)
        init_m = EMOTION_META[analysis["initial_emotion"]]
        final_m = EMOTION_META[analysis["final_emotion"]]
        dom_m = EMOTION_META[analysis["dominant_emotion"]]

        with m1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="stat-label">Total Turns</div>
                <div class="stat-value">{analysis["turns_count"]}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="stat-label">Emotion Changes</div>
                <div class="stat-value">{analysis["shifts_count"]}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="glass-card">
                <div class="stat-label">Initial → Final</div>
                <div class="stat-value" style="font-size:16px; margin-top:4px;">
                    <span style="color:{init_m['color']}">{init_m['icon']} {analysis['initial_emotion']}</span>
                    <span style="color:#64748b;">→</span>
                    <span style="color:{final_m['color']}">{final_m['icon']} {analysis['final_emotion']}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="glass-card">
                <div class="stat-label">Dominant Emotion</div>
                <div class="stat-value" style="color:{dom_m['color']}">{dom_m['icon']} {analysis["dominant_emotion"]}</div>
            </div>""", unsafe_allow_html=True)

        # 2. Interactive Plotly Trajectory Chart
        turns_data = analysis["turns"]
        xs = [f"T{t['turn']}: {t['speaker']}" for t in turns_data]
        vals = [t["valence"] for t in turns_data]
        cols = [t["color"] for t in turns_data]
        hover_text = [
            f"<b>{t['speaker']}</b> (Turn {t['turn']})<br>"
            f"Emotion: <b>{t['emotion']}</b> ({t['confidence']}% confidence)<br>"
            f"<i>\"{t['text']}\"</i>"
            for t in turns_data
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs,
            y=vals,
            mode="lines+markers",
            line=dict(color="rgba(56, 189, 248, 0.65)", width=3, shape="spline"),
            marker=dict(size=14, color=cols, line=dict(color="white", width=1.5)),
            hoverinfo="text",
            hovertext=hover_text,
        ))

        # Add neutral baseline reference
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text="Neutral Baseline", annotation_position="bottom right")

        fig.update_layout(
            title="Emotional Valence Trajectory Across Conversation",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Conversation Turns", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Emotional Valence", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)", zeroline=False),
            font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3. Turn-by-Turn Breakdown Feed
        st.markdown('<div class="section-title">🔍 Turn-by-Turn Emotional Progression</div>', unsafe_allow_html=True)
        for t in turns_data:
            shift_html = ""
            if t["transition"]["changed"]:
                prev_e = t["transition"]["from"]
                cur_e = t["transition"]["to"]
                if prev_e in EMOTION_META and cur_e in EMOTION_META:
                    pm = EMOTION_META[prev_e]
                    cm = EMOTION_META[cur_e]
                    shift_html = (
                        f'<div style="margin-top:8px; padding-top:6px; border-top:1px solid rgba(255,255,255,0.08); '
                        f'font-size:12.5px; color:#c4b5fd; display:flex; align-items:center; gap:6px;">'
                        f'⚡ <strong>Emotion Shift:</strong> '
                        f'<span style="color:{pm["color"]}">{pm["icon"]} {prev_e}</span> '
                        f'<span>→</span> '
                        f'<span style="color:{cm["color"]}">{cm["icon"]} {cur_e}</span> '
                        f'<span style="opacity:0.75">(Intensity: {t["transition"]["intensity"]})</span>'
                        f'</div>'
                    )

            card_html = (
                f'<div class="glass-card" style="margin-bottom:10px; border-left: 4px solid {t["color"]};">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">'
                f'<span style="font-weight:700; color:#f8fafc; font-size:14px;">Turn #{t["turn"]} · {t["speaker"]}</span>'
                f'<span class="bubble-emotion-badge" style="background:{t["color"]}22; color:{t["color"]};">'
                f'{t["icon"]} {t["emotion"]} ({t["confidence"]}%)'
                f'</span>'
                f'</div>'
                f'<div style="color:#e2e8f0; font-size:15px; line-height:1.4;">"{t["text"]}"</div>'
                f'{shift_html}'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)


def calculate_volatility(messages):
    if len(messages) < 2:
        return 0
    changes = 0
    for i in range(1, len(messages)):
        if messages[i]["emotion"] != messages[i - 1]["emotion"]:
            changes += 1
    return round(changes / (len(messages) - 1), 2)
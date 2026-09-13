"""
NEUROML-X : TRACE TOXIC AI
"AI That Detects. ML That Protects."

NVIDIA x Nebius Global AI Hackathon — Best Apps and Agents Track
Production-grade Cyberbullying Detection, Emotion Profiling, and Agentic Safety System.
Combines:
- Local Deep Learning Sequence Classification (toxic-bert + emotion-distilroberta on NVIDIA CUDA GPU)
- NVIDIA Nemotron Agentic Contextual Reasoning via Nebius Token Factory
- In-memory institutional PDF report generation with zero file locks
"""

import os
import sys
import json
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from utils.preprocessing import anonymize_text, normalize_text
from utils.risk_analysis import compute_combined_severity, get_risk_tier, RISK_DISCLAIMER
from utils.visualization import (
    create_toxicity_bar_chart,
    create_emotion_radar_chart,
    create_conversation_timeline_chart,
    create_risk_distribution_chart
)
from utils.reporting import generate_pdf_report
from services.ml_service import (
    load_ml_models,
    detect_toxicity_dl,
    detect_emotion_dl,
    detect_toxicity_fallback,
    detect_emotion_fallback,
    detect_categories,
    DEVICE,
    DEVICE_NAME
)
from services.nemotron_service import NemotronAgentService

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="NEUROML-X : TRACE TOXIC AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# MODERN DARK CYBER HACKATHON STYLING
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #090714 0%, #130f2b 50%, #0d0b1e 100%);
        color: #f1f5f9;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }

    .card {
        background: rgba(255, 255, 255, 0.035);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
        margin-bottom: 20px;
    }

    .card-metric {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 800;
        margin-top: 4px;
        margin-bottom: 2px;
    }

    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
    }

    .badge-chip {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        margin: 3px 5px 3px 0;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    .badge-chip-danger {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    .device-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11.5px;
        font-weight: 600;
        border: 1px solid rgba(16, 185, 129, 0.3);
        margin-bottom: 8px;
    }

    .nebius-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(139, 92, 246, 0.15);
        color: #c4b5fd;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11.5px;
        font-weight: 600;
        border: 1px solid rgba(139, 92, 246, 0.3);
        margin-bottom: 8px;
    }

    .alert-critical {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.9), rgba(185, 28, 28, 0.95));
        border: 2px solid #ef4444;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-weight: 800;
        font-size: 18px;
        color: #ffffff;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
        margin-bottom: 18px;
    }

    .alert-high {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.85), rgba(194, 65, 12, 0.9));
        border: 1px solid #f97316;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 17px;
        color: #ffffff;
        margin-bottom: 18px;
    }

    .alert-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.85), rgba(217, 119, 6, 0.9));
        border: 1px solid #f59e0b;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 16px;
        color: #ffffff;
        margin-bottom: 18px;
    }

    .alert-low {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.85), rgba(2, 132, 199, 0.9));
        border: 1px solid #38bdf8;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 16px;
        color: #ffffff;
        margin-bottom: 18px;
    }

    .alert-safe {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.85), rgba(5, 150, 105, 0.9));
        border: 1px solid #10b981;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 16px;
        color: #ffffff;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# SESSION STATE MANAGEMENT
# ------------------------------------------------------------
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "conversation_result" not in st.session_state:
    st.session_state.conversation_result = None
if "history" not in st.session_state:
    st.session_state.history = []
if "session_stats" not in st.session_state:
    st.session_state.session_stats = {
        "total": 0,
        "safe": 0,
        "toxic": 0,
        "cyberbullying": 0,
        "high_risk": 0,
        "severity_sum": 0.0
    }

# ------------------------------------------------------------
# MODEL & SERVICE BOOTSTRAP
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_ml_models():
    return load_ml_models()

with st.spinner("Initializing Deep Learning Engine & NVIDIA CUDA Hardware..."):
    toxic_tokenizer, toxic_model, emo_tokenizer, emo_model, models_loaded, load_err = get_ml_models()

# Load sample conversations
SAMPLE_CONVERSATIONS = {}
try:
    with open(os.path.join(CURRENT_DIR, "data", "sample_conversations.json"), "r") as f:
        SAMPLE_CONVERSATIONS = json.load(f).get("conversations", {})
except Exception:
    pass

SAMPLE_PRESETS = {
    "1. Severe Violent Threat": "I know where you live, watch your back because I will find you and beat you until you can't walk 😡🔪",
    "2. Cyber-Mobbing & Exclusion": "Nobody in this school likes you. Delete your account and leave our Discord server, you pathetic loser 😂🤡",
    "3. Identity Hate & Slurs": "You people don't belong here, go back to where you came from, we hate your disgusting kind 🤬",
    "4. Body Shaming & Humiliation": "Look in the mirror, you are hideous and fat. It is disgusting how ugly you are 🤮",
    "5. Gaming Lobby Rage (Slang)": "u r such an idiot, uninstall the game kys stfu noob gtfo 💀",
    "6. Benign Friendly Message": "Hey! Great job on the presentation today, everyone really loved your ideas! Keep it up 😊",
}

# ------------------------------------------------------------
# SIDEBAR CONTROLS & STATUS
# ------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=56)
    st.markdown("### System Telemetry")

    # Hardware Status
    if models_loaded:
        st.markdown(
            f"<div class='device-pill'>● NVIDIA Hardware: {DEVICE_NAME} ({DEVICE.upper()})</div>",
            unsafe_allow_html=True
        )
        st.caption("Deep Learning: `toxic-bert` + `emotion-distilroberta`")
    else:
        st.warning("⚠️ Running in Heuristic ML Fallback")
        if load_err:
            with st.expander("Diagnostic Info"):
                st.code(load_err)

    # Nebius Token Factory Agent Status
    nebius_key = os.getenv("NEBIUS_API_KEY", "").strip()
    user_key_input = st.text_input(
        "Nebius Token Factory API Key:",
        value=nebius_key,
        type="password",
        help="Paste your Nebius Token Factory key to activate NVIDIA Nemotron reasoning."
    )

    nemotron_service = NemotronAgentService(api_key=user_key_input)
    if nemotron_service.is_available:
        st.markdown(
            "<div class='nebius-pill'>● Nebius Agent: NVIDIA Nemotron 70B Active</div>",
            unsafe_allow_html=True
        )
        st.caption("Nebius Token Factory: Connected")
    else:
        st.markdown(
            "<div class='nebius-pill' style='background:rgba(234,179,8,0.15); color:#fde047; border-color:rgba(234,179,8,0.3);'>● Agent: Offline Fallback Engine</div>",
            unsafe_allow_html=True
        )
        st.caption("Using local heuristic agent reasoning.")

    st.markdown("---")
    st.markdown("### ⚙️ Privacy & Security Settings")
    show_anonymized = st.toggle("View Anonymized Text (PII-Redacted)", value=True)
    strict_threshold = st.toggle("Strict Bullying Threshold", value=False)

    st.markdown("---")
    st.markdown("### 🚨 Emergency Helpline Services")
    st.markdown("""
    - **National Cyber Crime Helpline:** `1930`
    - **Cyber Crime Reporting:** [cybercrime.gov.in](https://cybercrime.gov.in)
    - **Crisis Helpline (US):** `988` / Text `HOME` to `741741`
    - **Tele-MANAS:** `14416`
    """)

    if st.session_state.history:
        st.markdown("---")
        st.caption(f"Session Scans: {len(st.session_state.history)}")
        if st.button("🗑️ Clear Session History", use_container_width=True):
            st.session_state.history = []
            st.session_state.analysis_result = None
            st.session_state.conversation_result = None
            st.session_state.session_stats = {"total": 0, "safe": 0, "toxic": 0, "cyberbullying": 0, "high_risk": 0, "severity_sum": 0.0}
            st.rerun()

# ------------------------------------------------------------
# MAIN APPLICATION HEADER
# ------------------------------------------------------------
st.markdown("<h1 style='margin-bottom:0; color:#38bdf8;'>NEUROML-X</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:0; color:#e2e8f0; font-weight:600;'>TRACE TOXIC AI</h3>", unsafe_allow_html=True)
st.markdown("<p style='color:#a78bfa; font-size:16px; font-weight:700; letter-spacing:0.5px;'><i>\"AI That Detects. ML That Protects.\"</i></p>", unsafe_allow_html=True)
st.caption("NVIDIA × Nebius Global AI Hackathon • Best Apps and Agents Track")

# Navigation Tabs
tab_single, tab_conv, tab_dash, tab_explain, tab_about = st.tabs([
    "🔍 Single Message Analyzer",
    "💬 Conversation Analyzer",
    "📊 Risk Dashboard",
    "🧠 AI Explanation & Agent Trace",
    "ℹ️ About & Technology"
])

# ============================================================
# TAB 1: SINGLE MESSAGE ANALYZER
# ============================================================
with tab_single:
    st.markdown(
        """
        <div class='card'>
            <h4 style='margin-top:0; color:#38bdf8;'>🔒 Privacy-First Intelligent Triage</h4>
            <p style='margin-bottom:0; color:#cbd5e1; font-size:14px;'>
                Input communications are anonymized locally (PII redacted) and classified via local Deep Learning 
                (<code>toxic-bert</code> + <code>emotion-distilroberta</code>). The predictions are then contextually 
                interpreted by <b>NVIDIA Nemotron via Nebius Token Factory</b> for explainable intent and intervention guidance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_method = st.radio(
        "Choose Input Source:",
        ["Preset Scenarios & Manual Text", "Notification Simulator", "Upload File (.txt/.csv)"],
        horizontal=True,
    )

    input_text = ""

    if input_method == "Preset Scenarios & Manual Text":
        col_pre1, col_pre2 = st.columns([1, 2])
        with col_pre1:
            selected_preset = st.selectbox("Select Quick Test Scenario:", ["-- Choose a preset scenario --"] + list(SAMPLE_PRESETS.keys()))
        with col_pre2:
            st.caption("Pick a scenario above or enter your own custom text below.")

        default_val = SAMPLE_PRESETS[selected_preset] if selected_preset in SAMPLE_PRESETS else ""
        input_text = st.text_area(
            "Message to Evaluate:",
            value=default_val,
            height=120,
            placeholder="Example: You are stupid and nobody likes you. Stop talking to us.",
        )

    elif input_method == "Notification Simulator":
        col_sim1, col_sim2 = st.columns([1, 2])
        with col_sim1:
            sim_app = st.selectbox("Source Platform", ["WhatsApp", "Instagram Direct", "Discord", "Telegram", "SMS", "X (Twitter)"])
            sim_sender = st.text_input("Sender Handle", value="gamer_killer99")
        with col_sim2:
            sim_msg = st.text_area(
                "Simulated Push Notification Text",
                value="u r dead meat, we found your address and we are coming for you tonight 😡🔪",
                height=100
            )
        input_text = sim_msg

    else:
        uploaded_file = st.file_uploader("Upload Text or Chat Export (.txt, .csv)", type=["txt", "csv"])
        if uploaded_file:
            try:
                content = uploaded_file.read().decode("utf-8")
                lines = [l.strip() for l in content.splitlines() if l.strip()]
                if len(lines) > 1:
                    sel_line = st.selectbox("Select message line:", range(len(lines)), format_func=lambda i: f"Line {i+1}: {lines[i][:65]}...")
                    input_text = lines[sel_line]
                else:
                    input_text = content
            except Exception as e:
                st.error(f"Error reading file: {e}")

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        analyze_clicked = st.button("🚀 Analyze with ML + NVIDIA Nemotron", use_container_width=True, type="primary")
    with col_btn2:
        if st.button("🔄 Reset View", use_container_width=True):
            st.session_state.analysis_result = None
            st.rerun()

    if analyze_clicked:
        if not input_text.strip():
            st.warning("Please enter or select a message to analyze first.")
        else:
            with st.spinner("Executing ML Inference + NVIDIA Nemotron Agentic Reasoning..."):
                anonymized = anonymize_text(input_text)
                normalized = normalize_text(anonymized)

                # 1. Local ML Inference
                if models_loaded:
                    toxic_scores = detect_toxicity_dl(normalized, toxic_tokenizer, toxic_model)
                    top_emo, emo_conf, all_emotions = detect_emotion_dl(normalized, emo_tokenizer, emo_model)
                    ml_engine_name = f"unitary/toxic-bert & emotion-distilroberta ({DEVICE_NAME})"
                else:
                    toxic_scores = detect_toxicity_fallback(normalized)
                    top_emo, emo_conf, all_emotions = detect_emotion_fallback(normalized)
                    ml_engine_name = "Heuristic ML Safety Engine (Fallback)"

                categories = detect_categories(normalized, toxic_scores)

                # 2. NVIDIA Nemotron Agentic Reasoning via Nebius Token Factory
                nemotron_reasoning = nemotron_service.analyze_single_message(
                    text=normalized,
                    ml_scores=toxic_scores,
                    top_emotion=top_emo,
                    emotion_confidence=emo_conf,
                    detected_categories=categories
                )

                # 3. Unified Severity Computation (0 - 100)
                severity = compute_combined_severity(toxic_scores, all_emotions, nemotron_reasoning)
                tier_info = get_risk_tier(severity)

                # Store result
                st.session_state.analysis_result = {
                    "raw_text": input_text,
                    "anonymized_text": anonymized,
                    "normalized_text": normalized,
                    "toxic_scores": toxic_scores,
                    "top_emo": top_emo,
                    "emo_conf": emo_conf,
                    "all_emotions": all_emotions,
                    "categories": categories,
                    "nemotron": nemotron_reasoning,
                    "severity": severity,
                    "tier_info": tier_info,
                    "ml_engine_name": ml_engine_name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # Update Session Statistics
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "text": input_text[:45] + ("..." if len(input_text) > 45 else ""),
                    "severity": severity,
                    "classification": nemotron_reasoning.get("classification", "SAFE"),
                    "tier": tier_info["tier"]
                })
                stats = st.session_state.session_stats
                stats["total"] += 1
                stats["severity_sum"] += severity
                cls_val = nemotron_reasoning.get("classification", "SAFE")
                if cls_val == "SAFE":
                    stats["safe"] += 1
                elif cls_val in ["TOXIC", "POSSIBLY TOXIC"]:
                    stats["toxic"] += 1
                elif cls_val in ["CYBERBULLYING", "SEVERE HARASSMENT"]:
                    stats["cyberbullying"] += 1
                if severity >= 61.0:
                    stats["high_risk"] += 1

    # Render Persistent Results
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        sev = res["severity"]
        tier = res["tier_info"]
        nemo = res["nemotron"]

        st.markdown("---")

        # Dynamic Risk Alert Banner
        st.markdown(
            f"""
            <div class="{tier['badge_class']}">
                🛡️ {tier['tier'].upper()} IDENTIFIED: {nemo.get('classification', 'SAFE')}<br/>
                <span style="font-size:13.5px; font-weight:normal;">{tier['recommendation']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if show_anonymized:
            with st.expander("👁️ Inspect Sanitized & PII-Redacted Text", expanded=False):
                col_s1, col_s2 = st.columns(2)
                col_s1.caption("Original Raw Message:")
                col_s1.code(res["raw_text"])
                col_s2.caption("Sanitized & PII-Redacted Message:")
                col_s2.code(res["anonymized_text"])

        # High-Level Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(
            f"""
            <div class='card-metric'>
                <div class='metric-label'>Combined Severity</div>
                <div class='metric-value' style='color:{tier["color"]};'>{sev:.1f}<span style='font-size:15px;'>/100</span></div>
                <div style='font-size:11px; color:#94a3b8;'>Tier: {tier['tier']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        top_tox_label = max(res["toxic_scores"], key=res["toxic_scores"].get)
        top_tox_val = res["toxic_scores"][top_tox_label]
        k2.markdown(
            f"""
            <div class='card-metric'>
                <div class='metric-label'>Peak ML Toxicity</div>
                <div class='metric-value' style='color:#38bdf8;'>{top_tox_val:.1f}%</div>
                <div style='font-size:11px; color:#94a3b8;'>{top_tox_label.replace('_', ' ').title()}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        emo_icons = {"anger": "😡", "disgust": "🤢", "fear": "😨", "joy": "😊", "neutral": "😐", "sadness": "😢", "surprise": "😲"}
        k3.markdown(
            f"""
            <div class='card-metric'>
                <div class='metric-label'>Primary Emotion</div>
                <div class='metric-value' style='color:#f472b6;'>{emo_icons.get(res['top_emo'].lower(), '🎯')} {res['top_emo'].title()}</div>
                <div style='font-size:11px; color:#94a3b8;'>{res['emo_conf']*100:.1f}% Confidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        k4.markdown(
            f"""
            <div class='card-metric'>
                <div class='metric-label'>Nemotron Confidence</div>
                <div class='metric-value' style='color:#a78bfa;'>{nemo.get('nemotron_confidence', 0.95)*100:.0f}%</div>
                <div style='font-size:11px; color:#94a3b8;'>Agent Agreement</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br/>", unsafe_allow_html=True)

        # Agent Reasoning & ML Visuals Row
        col_nemotron, col_charts = st.columns([1.1, 0.9])

        with col_nemotron:
            st.markdown("### 🤖 NVIDIA Nemotron Agentic Reasoning")
            st.markdown(
                f"""
                <div class='card'>
                    <div style='font-size:12px; color:#a78bfa; margin-bottom:8px; font-weight:700;'>
                        ENGINE: {nemo.get('engine', 'NVIDIA Nemotron')}
                    </div>
                    <p><b>Contextual Intent:</b> {nemo.get('context_intent', 'N/A')}</p>
                    <p><b>Explanation:</b> {nemo.get('explanation', 'N/A')}</p>
                    <div style='margin-top:10px;'>
                        <b>Detected Signals:</b><br/>
                        {''.join([f"<span class='badge-chip'>• {s}</span>" for s in nemo.get('detected_signals', [])])}
                    </div>
                    <div style='margin-top:14px;'>
                        <b>🛡️ Recommended Protective Action:</b><br/>
                        <span style='color:#38bdf8;'>{nemo.get('recommended_action', 'None')}</span>
                    </div>
                    <div style='margin-top:14px; background:rgba(255,255,255,0.03); padding:10px; border-radius:8px; border-left:3px solid #34d399;'>
                        <b>💬 Suggested De-escalating Victim Response:</b><br/>
                        <i>"{nemo.get('safe_victim_response', '')}"</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 🎯 Flagged Cyberbullying Categories")
            if res["categories"] == ["None (Benign)"]:
                st.success("✅ No overt cyberbullying categories flagged.")
            else:
                cat_chips = "".join([f"<span class='badge-chip badge-chip-danger'>⚠️ {c}</span>" for c in res["categories"]])
                st.markdown(f"<div>{cat_chips}</div>", unsafe_allow_html=True)

        with col_charts:
            st.markdown("### 🧪 Deep Learning Feature Breakdown")
            fig_bar = create_toxicity_bar_chart(res["toxic_scores"])
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("### 🎭 Emotion Distribution Radar")
            fig_radar = create_emotion_radar_chart(res["all_emotions"])
            st.plotly_chart(fig_radar, use_container_width=True)

        # PDF Download Section
        st.markdown("---")
        st.markdown("### 📄 Formal Digital Safety Incident Report")
        st.caption("Generate an official, court/school-admissible PDF document combining ML features with Nemotron agent reasoning.")

        pdf_bytes = generate_pdf_report(
            raw_text=res["raw_text"],
            anonymized_text=res["anonymized_text"],
            toxicity_scores=res["toxic_scores"],
            top_emotion=res["top_emo"],
            emotion_confidence=res["emo_conf"],
            all_emotions=res["all_emotions"],
            categories=res["categories"],
            severity=sev,
            nemotron_reasoning=nemo,
            ml_engine_name=res["ml_engine_name"]
        )

        st.download_button(
            label="📥 Download Official Incident PDF Report",
            data=pdf_bytes,
            file_name=f"NeuroMLX_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ============================================================
# TAB 2: CONVERSATION ANALYZER
# ============================================================
with tab_conv:
    st.markdown(
        """
        <div class='card'>
            <h4 style='margin-top:0; color:#38bdf8;'>💬 Multi-Turn Conversation Analyzer</h4>
            <p style='margin-bottom:0; color:#cbd5e1; font-size:14px;'>
                Evaluate multi-turn group dialogues or private message threads. The system tracks 
                turn-by-turn toxicity with the ML classifier while <b>NVIDIA Nemotron</b> reasons over 
                repeated targeting, mobbing dynamics, and hostility escalation trends.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    conv_preset = st.selectbox(
        "Choose a Preset Dialogue Scenario:",
        ["-- Load a preset dialogue --"] + list(SAMPLE_CONVERSATIONS.keys())
    )

    raw_conv_input = ""
    if conv_preset in SAMPLE_CONVERSATIONS:
        preset_dialogue = SAMPLE_CONVERSATIONS[conv_preset]
        raw_conv_input = "\n".join([f"{turn['speaker']}: {turn['text']}" for turn in preset_dialogue])

    conv_text = st.text_area(
        "Enter or Edit Multi-Turn Transcript (Format: 'Speaker: Message' per line):",
        value=raw_conv_input,
        height=180,
        placeholder="Alex: Hey did you finish the slides?\nJordan: Nobody asked you Alex. Why are you even here?"
    )

    if st.button("🔍 Analyze Conversation Dynamics", type="primary", use_container_width=True):
        lines = [l.strip() for l in conv_text.splitlines() if l.strip()]
        parsed_turns = []
        for l in lines:
            if ":" in l:
                spk, msg = l.split(":", 1)
                parsed_turns.append({"speaker": spk.strip(), "text": msg.strip()})
            else:
                parsed_turns.append({"speaker": "User", "text": l.strip()})

        if not parsed_turns:
            st.warning("Please enter conversation turns in 'Speaker: Message' format.")
        else:
            with st.spinner("Processing turn-by-turn ML inference & Nemotron conversation analysis..."):
                turn_scores = []
                for turn in parsed_turns:
                    norm = normalize_text(anonymize_text(turn["text"]))
                    if models_loaded:
                        tox = detect_toxicity_dl(norm, toxic_tokenizer, toxic_model)
                        emo, conf, all_e = detect_emotion_dl(norm, emo_tokenizer, emo_model)
                    else:
                        tox = detect_toxicity_fallback(norm)
                        emo, conf, all_e = detect_emotion_fallback(norm)
                    
                    peak = max(tox.values()) if tox else 0.0
                    turn_scores.append({
                        "peak_toxicity": peak,
                        "top_emotion": emo,
                        "severity": round((peak * 0.7) + (30.0 if emo in ["anger", "disgust"] else 0.0), 1)
                    })

                # Nemotron multi-turn evaluation
                conv_analysis = nemotron_service.analyze_conversation(parsed_turns, turn_scores)
                st.session_state.conversation_result = {
                    "turns": parsed_turns,
                    "scores": turn_scores,
                    "analysis": conv_analysis
                }

    if st.session_state.conversation_result:
        c_res = st.session_state.conversation_result
        turns = c_res["turns"]
        scores = c_res["scores"]
        analysis = c_res["analysis"]

        st.markdown("---")
        st.markdown("### 📈 Dialogue Escalation Timeline")
        fig_conv = create_conversation_timeline_chart(turns, scores)
        st.plotly_chart(fig_conv, use_container_width=True)

        col_ca1, col_ca2 = st.columns(2)
        with col_ca1:
            st.markdown("### 🤖 Agentic Conversation Assessment")
            cls_color = "#ef4444" if analysis.get("overall_classification") in ["SEVERE HARASSMENT", "CYBERBULLYING"] else "#38bdf8"
            st.markdown(
                f"""
                <div class='card'>
                    <div style='font-size:14px; font-weight:700; color:{cls_color};'>
                        CLASSIFICATION: {analysis.get('overall_classification', 'SAFE')}
                    </div>
                    <p><b>Summary:</b> {analysis.get('summary', 'N/A')}</p>
                    <p><b>Escalation Dynamics:</b> {analysis.get('escalation_details', 'N/A')}</p>
                    <p><b>Recommended Action:</b> <span style='color:#38bdf8;'>{analysis.get('recommended_intervention', 'N/A')}</span></p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_ca2:
            st.markdown("### 👥 Role & Pattern Identification")
            st.markdown(
                f"""
                <div class='card'>
                    <p><b>Escalation Detected:</b> {'🚨 YES' if analysis.get('escalation_detected') else '✅ NO'}</p>
                    <p><b>Repeated Targeting:</b> {'⚠️ YES' if analysis.get('repeated_targeting') else '✅ NO'}</p>
                    <p><b>Identified Aggressor:</b> <span style='color:#fca5a5;'>{analysis.get('primary_aggressor') or 'None identified'}</span></p>
                    <p><b>Targeted Participant:</b> <span style='color:#93c5fd;'>{analysis.get('targeted_user') or 'None identified'}</span></p>
                </div>
                """,
                unsafe_allow_html=True
            )

# ============================================================
# TAB 3: RISK DASHBOARD
# ============================================================
with tab_dash:
    st.markdown("### 📊 Real-Time Safety & Risk Telemetry")
    stats = st.session_state.session_stats
    avg_sev = (stats["severity_sum"] / stats["total"]) if stats["total"] > 0 else 0.0

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("Total Scans", stats["total"])
    d2.metric("Safe Messages", stats["safe"])
    d3.metric("Toxic Flagged", stats["toxic"])
    d4.metric("Cyberbullying", stats["cyberbullying"])
    d5.metric("High/Critical", stats["high_risk"])
    d6.metric("Avg Severity", f"{avg_sev:.1f}")

    st.markdown("<br/>", unsafe_allow_html=True)
    col_chart_d, col_table_d = st.columns([1, 1.2])

    with col_chart_d:
        fig_dist = create_risk_distribution_chart(st.session_state.history)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_table_d:
        st.markdown("#### Recent Scanned Messages")
        if not st.session_state.history:
            st.info("No messages analyzed in this session yet. Run a single or conversation scan to populate telemetry.")
        else:
            df_hist = pd.DataFrame(reversed(st.session_state.history))
            st.dataframe(df_hist, use_container_width=True, hide_index=True)

# ============================================================
# TAB 4: AI EXPLANATION & AGENT TRACE
# ============================================================
with tab_explain:
    st.markdown("### 🧠 Explainable AI: The Hybrid ML + Nemotron Architecture")
    st.markdown(
        """
        <div class='card'>
            <h4 style='color:#38bdf8;'>Why Combine Machine Learning with NVIDIA Nemotron?</h4>
            <div style='display:grid; grid-template-columns: 1fr 1fr; gap: 16px;'>
                <div style='background:rgba(255,255,255,0.03); padding:16px; border-radius:10px; border-left:3px solid #3b82f6;'>
                    <h5 style='color:#93c5fd; margin-top:0;'>1. Why Machine Learning is Used</h5>
                    <ul style='font-size:13.5px; color:#cbd5e1; padding-left:18px;'>
                        <li><b>Sub-300ms Deterministic Inference:</b> Runs directly on user's NVIDIA GPU via PyTorch CUDA.</li>
                        <li><b>Objective Feature Quantization:</b> Produces exact calibrated probability vectors across 6 toxicity and 7 emotion dimensions.</li>
                        <li><b>Zero Cost & Offline Resilience:</b> Operates without cloud token consumption or network dependence.</li>
                    </ul>
                </div>
                <div style='background:rgba(255,255,255,0.03); padding:16px; border-radius:10px; border-left:3px solid #8b5cf6;'>
                    <h5 style='color:#c4b5fd; margin-top:0;'>2. Why NVIDIA Nemotron is Used</h5>
                    <ul style='font-size:13.5px; color:#cbd5e1; padding-left:18px;'>
                        <li><b>Contextual Disambiguation:</b> Distinguishes between friendly gaming banter and malicious harassment.</li>
                        <li><b>Interpersonal Dynamics:</b> Identifies gaslighting, coercion, and multi-turn escalation trends.</li>
                        <li><b>Agentic Intervention:</b> Generates safe de-escalation responses tailored to the victim.</li>
                    </ul>
                </div>
            </div>
            <div style='margin-top:16px; font-size:14px; color:#cbd5e1;'>
                <b>Nebius Token Factory Role:</b> Provides enterprise high-throughput API endpoints for 
                <code>nvidia/Llama-3.1-Nemotron-70B-Instruct-HF</code>, eliminating the need to download 
                multi-gigabyte weights to client devices while ensuring ultra-low latency inference.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.analysis_result:
        st.markdown("#### Live Agent Trace (Latest Scan)")
        st.json({
            "ml_probabilities": st.session_state.analysis_result["toxic_scores"],
            "emotion": st.session_state.analysis_result["top_emo"],
            "nemotron_reasoning": st.session_state.analysis_result["nemotron"],
            "severity_index": st.session_state.analysis_result["severity"]
        })

# ============================================================
# TAB 5: ABOUT & TECHNOLOGY
# ============================================================
with tab_about:
    st.markdown("### ℹ️ About NEUROML-X : TRACE TOXIC AI")
    st.markdown(
        """
        <div class='card'>
            <h4 style='color:#38bdf8;'>NVIDIA × Nebius Global AI Hackathon</h4>
            <p><b>Target Track:</b> Best Apps and Agents Track</p>
            <p>
                Cyberbullying and toxic online mobbing have severe psychological consequences. 
                Traditional rule-based filters either fail to catch nuanced harassment or over-censor innocent slang. 
                <b>NEUROML-X</b> solves this with a hybrid pipeline: local GPU-accelerated deep learning handles 
                instant quantitative scoring, while <b>NVIDIA Nemotron on Nebius Token Factory</b> provides deep 
                contextual reasoning, explainability, and proactive digital safety guidance.
            </p>
            <h5 style='color:#38bdf8;'>Technology Stack</h5>
            <ul style='font-size:14px; color:#cbd5e1;'>
                <li><b>Hardware Acceleration:</b> NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.1)</li>
                <li><b>Local Deep Learning:</b> PyTorch 2.5, Hugging Face Transformers (<code>unitary/toxic-bert</code>, <code>j-hartmann/emotion-english-distilroberta-base</code>)</li>
                <li><b>Generative AI Agent:</b> NVIDIA Nemotron 70B (<code>nvidia/Llama-3.1-Nemotron-70B-Instruct-HF</code>) via Nebius Token Factory</li>
                <li><b>Interactive UI:</b> Streamlit, Plotly, Custom Dark Cyber CSS</li>
                <li><b>Reporting:</b> ReportLab In-Memory PDF Engine (tamper-evident, zero disk locks)</li>
            </ul>
            <h5 style='color:#ef4444;'>Safety & Ethical Disclaimers</h5>
            <p style='font-size:13px; color:#94a3b8;'>
                """ + RISK_DISCLAIMER + """
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.caption("NEUROML-X : TRACE TOXIC AI • 'AI That Detects. ML That Protects.' • NVIDIA × Nebius Global AI Hackathon")

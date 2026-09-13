# NEUROML-X : TRACE TOXIC AI
> **"AI That Detects. ML That Protects."**

[![Hackathon](https://img.shields.io/badge/Hackathon-NVIDIA%20%C3%97%20Nebius%20Global%20AI-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://nebius.com)
[![Track](https://img.shields.io/badge/Track-Best%20Apps%20%26%20Agents-8B5CF6?style=for-the-badge)](https://nebius.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%20CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![NVIDIA Nemotron](https://img.shields.io/badge/NVIDIA-Nemotron--70B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF)
[![Nebius Token Factory](https://img.shields.io/badge/Nebius-Token%20Factory-6366F1?style=for-the-badge)](https://nebius.ai)

---

## 📌 Elevator Pitch

**NEUROML-X TRACE TOXIC AI** is a dual-engine digital safety system that combines local **NVIDIA CUDA-accelerated Machine Learning** with **NVIDIA Nemotron 70B via Nebius Token Factory**. While local deep learning models instantly quantify toxicity and emotional affect in sub-300ms, the remote Nemotron agent analyzes interpersonal intent, detects multi-turn harassment escalation, and equips victims with safe, de-escalating intervention actions.

---

## 🛑 The Problem

Online harassment and cyberbullying have reached epidemic levels across social networks, gaming lobbies, and messaging platforms:
- **Subtle Aggression is Missed:** Traditional keyword blocklists fail against sarcasm, passive-aggressive mobbing, and evolving internet slang.
- **Over-Censorship:** Naive filters flag civil debates or common gaming terminology as offensive, eroding user trust.
- **Lack of Actionable Help:** Current moderation systems simply hide comments without helping the victim understand what happened or providing safe response advice.
- **Latency vs Context Trade-off:** Large Language Models are too slow and expensive to evaluate every raw notification, while lightweight rule systems lack the context to understand human intent.

---

## 💡 The Solution: Hybrid ML + Agent Architecture

NEUROML-X solves this through a **two-stage hybrid intelligence pipeline**:
1. **Stage 1 (ML First-Pass):** Lightweight, GPU-accelerated sequence classification models (`toxic-bert` + `emotion-distilroberta`) evaluate the raw text locally on the device in **<300ms**, generating exact probability distributions across 6 toxicity dimensions and 7 emotion profiles.
2. **Stage 2 (NVIDIA Nemotron via Nebius Token Factory):** The structured probabilities and dialogue context are handed to **NVIDIA Nemotron 70B** through the **Nebius Token Factory**. Nemotron reasons over the relational dynamic, evaluates escalation trends across turns, identifies who is being targeted, and formulates safe, non-retaliatory intervention strategies.

---

## 🔬 Why Machine Learning + Why Nemotron?

| Objective | ⚡ Local Machine Learning (`toxic-bert` / `distilroberta`) | 🧠 NVIDIA Nemotron 70B via Nebius Token Factory |
|---|---|---|
| **Role** | First-Pass Quantitative Classification | Second-Pass Contextual Reasoning & Agentic Safety |
| **Inference Time** | **<300ms** (runs locally on NVIDIA GPU via CUDA) | **~1-2s** (high-throughput cloud token stream) |
| **What It Detects** | Toxicity, insults, obscenity, physical threats, emotion intensity | Relational intent, sarcasm, gaslighting, mobbing trends, harassment patterns |
| **Output** | Exact probability vectors (0.0% – 100.0%) | Explainable signals, root-cause narrative, victim response options |
| **Cost & Network** | Free, zero token cost, runs offline on laptop | Low-cost API via Nebius Token Factory, called when context is needed |

### Where Nemotron is Used & How Nebius Token Factory Enabled It
- **Intent Disambiguation:** When local ML detects elevated toxicity, Nemotron is prompted with the raw text alongside the ML probability vector to determine whether the communication is playful banter between friends or genuine harassment.
- **Multi-Turn Conversation Dynamics:** In conversation mode, Nemotron inspects the sequence of turns to detect escalating aggression and repeated targeting.
- **Safe Victim Communication:** Nemotron generates assertive, de-escalating responses for the victim so they can set boundaries without triggering retaliatory violence.
- **Nebius Token Factory Role:** Nebius provides high-speed, enterprise-grade OpenAI-compatible API endpoints for `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF`. This allows NEUROML-X to tap into a 70-billion-parameter reasoning model without requiring 140GB+ of local VRAM or downloading massive weights to personal laptops.

---

## 🏗️ System Architecture

```
                                USER INPUT / CHAT TRANSCRIPT
                                             ↓
                            [ Preprocessing & Privacy Layer ]
                         (PII Redaction, Slang & Emoji Mapping)
                                             ↓
                     [ Local Deep Learning Sequence Classifier ]
                     (NVIDIA RTX 3050 CUDA / toxic-bert + roberta)
                                             ↓
                          ML TOXICITY & EMOTION PROBABILITY VECTORS
                                             ↓
                              [ Context & Agent Router ]
                                             ↓
                       [ NVIDIA Nemotron 70B (Nebius Token Factory) ]
                         • Context Analysis    • Intent Evaluation
                         • Escalation Dynamics • Action Recommendation
                                             ↓
                              [ Unified Severity Score (0-100) ]
                             (5 Tiers: Safe, Low, Moderate, High, Critical)
                                             ↓
                      [ NEUROML-X STREAMLIT HACKATHON DASHBOARD ]
               ┌───────────────────────────┼───────────────────────────┐
               ↓                           ↓                           ↓
      Single Message Mode        Conversation Analyzer         Risk Dashboard
```

---

## ✨ Key Features

1. **🔒 PII Anonymization & Privacy Preserving:** Phone numbers, email addresses, external links, and social handles are redacted *before* neural processing.
2. **⚡ NVIDIA CUDA Hardware Acceleration:** Deep learning sequence classifiers leverage local NVIDIA Tensor Cores for sub-300ms inference.
3. **🤖 Nebius Token Factory Integration:** Direct OpenAI-compatible client connection to `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` with automatic local fallback if offline.
4. **💬 Multi-Turn Conversation Analyzer:** Evaluates chat transcripts turn-by-turn with an interactive timeline chart tracking hostility escalation.
5. **🎯 Explainable AI (XAI):** Every alert includes peak toxicity, primary emotion, detected signals, plain-language reasoning, and safety advice.
6. **📊 Multi-Tiered Severity Index (0–100):**
   - **0–20:** Safe
   - **21–40:** Low Risk
   - **41–60:** Moderate Risk
   - **61–80:** High Risk
   - **81–100:** Critical Risk
7. **📄 Institutional PDF Reports:** In-memory ReportLab PDF generator produces court/school-admissible incident reports with zero disk locks.

---

## 🛠️ Technology Stack

- **GPU Acceleration:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.1)
- **Frameworks:** PyTorch 2.5, Hugging Face Transformers (`unitary/toxic-bert`, `j-hartmann/emotion-english-distilroberta-base`)
- **LLM Reasoning:** NVIDIA Nemotron 70B (`nvidia/Llama-3.1-Nemotron-70B-Instruct-HF`)
- **Cloud Infrastructure:** Nebius Token Factory (Nebius Studio API)
- **Frontend / UI:** Streamlit with custom Dark Cyber aesthetic & Plotly charts
- **Reporting Engine:** ReportLab In-Memory PDF Engine
- **Testing:** Python `unittest` suite (8/8 automated test cases passing)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/neuroml-x-trace-toxic-ai.git
cd neuroml-x-trace-toxic-ai
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` to include your Nebius Token Factory API key:
```env
NEBIUS_API_KEY=your_actual_nebius_key_here
NEBIUS_BASE_URL=https://api.studio.nebius.ai/v1
NEMOTRON_MODEL=nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
```
> **Note:** If `NEBIUS_API_KEY` is omitted, NEUROML-X will seamlessly activate its **Local Agent Fallback Engine** so judges and users can still test all workflows without credentials!

### 4. Run the Application
```bash
streamlit run app.py
```
Or on Windows, simply double-click:
```cmd
run_app.bat
```
Navigate to `http://localhost:8501` in your browser.

---

## 🧪 Running Automated Tests

A comprehensive unit test suite is included in `tests/test_core.py`:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
Test coverage verifies:
- Safe benign communication
- Direct toxic text detection
- Severe cyberbullying & physical threats
- Empty & whitespace input resilience
- Long-input truncation & memory handling
- Nemotron API offline fallback
- Invalid API key graceful degradation
- Multi-turn conversation escalation tracking

---

## 🖥️ Application Walkthrough

### 1. Single Message Analyzer
Select from 6 built-in presets (e.g. violent threats, cyber-mobbing, identity hate, gaming rage, or benign messages) or test your own text. View peak toxicity, emotional radar breakdown, Nemotron contextual reasoning, and download an official incident PDF report.

### 2. Conversation Analyzer
Paste multi-line dialogue transcripts. Observe turn-by-turn ML scoring and an interactive escalation timeline chart showing how aggression develops over time.

### 3. Risk Dashboard
Monitor aggregate session metrics including total scans, safe vs toxic distribution, average severity score, and recent interaction history.

### 4. AI Explanation & Agent Trace
Inspect live raw JSON traces sent between local deep learning models and NVIDIA Nemotron via Nebius Token Factory.

---

## ⚠️ Safety, Ethical Guidelines & Disclaimers

- **Non-Diagnostic:** NEUROML-X is an automated digital safety screening aid. It does not provide psychiatric diagnoses or legal determinations.
- **Privacy Guarantee:** Input text is sanitized locally to strip phone numbers, email addresses, handles, and URLs.
- **Non-Retaliation Principle:** Suggested responses prioritize de-escalation and assertive boundary-setting rather than aggressive retaliation.

---

## 🏆 Hackathon Submission Details

- **Hackathon:** NVIDIA × Nebius Global AI Hackathon
- **Track:** Best Apps and Agents Track
- **Submission Title:** NEUROML-X : TRACE TOXIC AI
- **Tagline:** *"AI That Detects. ML That Protects."*

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

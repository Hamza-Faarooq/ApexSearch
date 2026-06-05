# 🏎️ ApexSearch: Audio-Visual RAG & Multimodal Video Retrieval Engine

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-F9AB00.svg)](https://huggingface.co/)
[![Groq API](https://img.shields.io/badge/Groq-Llama%203-black.svg)](https://groq.com/)

*Finding a specific 5-second crash in a 2-hour Formula 1 broadcast is a nightmare. So, I built a multimodal AI system that finds it for you, listens to the engines, and commentates on the event in real time.*

---

## 🏁 The Pitch
Standard video classifiers just output static labels (e.g., "Crash: 98%"). I wanted to build a **true retrieval engine** that actually *understands* both the sights and sounds of a high-speed sports broadcast. 

**ApexSearch** is a Retrieval-Augmented Generation (RAG) pipeline built for video. You type a natural language query like *"Mechanics rushing to change tires"* or *"Loud engine revving and a crash"*, and the system mathematically weights visual and audio vectors to find the exact timestamp. Then, it passes the raw data to a Vision-Language pipeline to generate hyped-up, David Croft-style live commentary.

## 🧠 How It Works Under the Hood

The system architecture is broken down into a **Two-Agent Decoupled Pipeline**, designed specifically to handle heavy foundational models while maintaining blazing-fast retrieval latency.

### 1️⃣ Dual-Sensory Data Ingestion & Indexing
* **Visual:** We strip the broadcast down to 1 Frame Per Second (saving 97% of redundant compute) and map the frames into a 512-dimensional vector space using OpenAI’s **CLIP** (Vision Transformer).
* **Audio:** We slice the broadcast audio into 1-second `.wav` chunks and map them using LAION’s **CLAP** (Contrastive Language-Audio Pretraining).
* Both sets of vectors undergo **L2 Normalization**, converting heavy Cosine Similarity calculations into lightning-fast Inner Product matrix multiplications inside **FAISS**.

### 2️⃣ Late-Fusion Retrieval Engine
When a user searches for *"tires screeching"*, the text is vectorized by both CLIP and CLAP. 
The system runs a **dynamically weighted late-fusion scoring heuristic**:
`Score = (α * Visual_Distance) + ((1 - α) * Audio_Distance)`
This mathematical fusion resolves cross-modal ambiguity—for instance, trusting the audio over the visual feed during heavy motion blur or rain spray.

### 3️⃣ Agentic VLM Commentary (The Two-Agent Flow)
To get high-energy F1 commentary without hallucinating (e.g., confusing an F1 car for a motorcycle), tasks are decoupled:
* **The Observer (Groq Vision / Llama-3.2-11B-Vision):** Acts purely clinically. Looks at the retrieved frames and extracts a rigid JSON payload (`{"event_type": "pit_stop", "intensity": "high"}`).
* **The Commentator (Groq / Llama-3-70B):** Takes the JSON and a heavy System Prompt, breaking standard AI politeness to generate pure, adrenaline-pumping live commentary mimicking F1 broadcasting legends.

---

## 🛠️ Tech Stack
* **Core ML:** PyTorch, FAISS (Facebook AI Similarity Search), OpenCV, Librosa, MoviePy
* **Foundational Models:** CLIP (`openai/clip-vit-base-patch32`), CLAP (`laion/clap-htsat-unfused`)
* **LLM / VLM API:** Groq (Llama-3-70B & Llama-3.2-11B-Vision)
* **Frontend & Deployment:** Streamlit, Streamlit Community Cloud

---

## 🚀 Quick Start (Run it Locally)

Want to run ApexSearch on your own machine? It’s designed to be completely dynamically ingested.

### 1. Clone the repository

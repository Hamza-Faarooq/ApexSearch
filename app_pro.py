import streamlit as st
import cv2
import os
import faiss
import pickle
import torch
import gc
import librosa
import numpy as np
import base64
from PIL import Image
from moviepy.editor import VideoFileClip
from transformers import CLIPProcessor, CLIPModel, ClapProcessor, ClapModel
from groq import Groq

# --- Page Setup ---
st.set_page_config(page_title="ApexSearch Pro", layout="wide", page_icon="🏎️")
st.title("🏎️ ApexSearch Pro: Multi-Modal F1 RAG Platform")

# Paths
UPLOAD_DIR = "custom_upload"
CUSTOM_FRAMES_DIR = os.path.join(UPLOAD_DIR, "frames")
DEFAULT_FRAMES_DIR = "data/frames"

os.makedirs(CUSTOM_FRAMES_DIR, exist_ok=True)

device = "cpu" # FORCED TO CPU FOR CLOUD DEPLOYMENT
groq_client = Groq() # Reads GROQ_API_KEY from environment/secrets

# --- Agent 1: The Observer (Groq Vision API) ---
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_clinical_json_via_api(image_path):
    base64_image = encode_image_to_base64(image_path)
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this frame from an F1 stream. Output ONLY a raw clinical JSON string with keys: 'event_type', 'entities_involved', 'intensity'. No conversational padding."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        return '{"event_type": "unknown", "entities_involved": ["car", "track"], "intensity": "medium"}'

# --- Agent 2: The Commentator (Groq Text API) ---
def generate_croft_commentary(clinical_json):
    system_prompt = """You are the world’s most energetic, passionate, and iconic Formula 1 television commentator David Croft. 
    Translate the clinical JSON payload of a race scene into exactly ONE loud, thrilling sentence of live commentary. 
    Do NOT use AI language like 'The JSON shows' or 'In the image'. Just scream the action using heavy F1 terminology!"""
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": clinical_json}],
            temperature=0.85,
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        return "AND IT'S UNBELIEVABLE SCENES ACROSS THE TRACK!"

# --- Encoders for Local Search ---
@st.cache_resource
def load_encoders():
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    clap_proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    return clip, clip_proc, clap, clap_proc

# --- Dynamic Processing for Custom Uploads ---
def process_custom_video(video_path):
    clip_model, clip_processor, clap_model, clap_processor = load_encoders()
    v_embeddings, a_embeddings = [], []
    v_names, a_names = [], []
    
    # Extract Frames (1 FPS)
    cam = cv2.VideoCapture(video_path)
    fps_native = cam.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(fps_native)))
    frame_idx = 0
    success, frame = cam.read()
    
    while success:
        if frame_idx % frame_interval == 0:
            sec = frame_idx // frame_interval
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            img_name = f"custom_frame_{sec:04d}.jpg"
            pil_img.save(os.path.join(CUSTOM_FRAMES_DIR, img_name))
            
            inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = clip_model.get_image_features(**inputs)
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
            v_embeddings.append(feat.cpu().numpy())
            v_names.append(img_name)
        success, frame = cam.read()
        frame_idx += 1
    cam.release()
    
    # Extract Audio
    video = VideoFileClip(video_path)
    temp_audio = os.path.join(UPLOAD_DIR, "temp_audio.wav")
    video.audio.write_audiofile(temp_audio, fps=48000, logger=None)
    y, sr = librosa.load(temp_audio, sr=48000)
    duration = int(librosa.get_duration(y=y, sr=sr))
    
    for sec in range(duration):
        start_sample = sec * sr
        end_sample = (sec + 1) * sr
        chunk = y[start_sample:end_sample]
        if len(chunk) < sr / 2: continue
        
        inputs = clap_processor(audios=chunk, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = clap_model.get_audio_features(**inputs)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        a_embeddings.append(feat.cpu().numpy())
        a_names.append(f"custom_frame_{sec:04d}.wav")
        
    v_matrix = np.vstack(v_embeddings).astype(np.float32)
    a_matrix = np.vstack(a_embeddings).astype(np.float32)
    
    v_idx = faiss.IndexFlatIP(v_matrix.shape[1])
    v_idx.add(v_matrix)
    a_idx = faiss.IndexFlatIP(a_matrix.shape[1])
    a_idx.add(a_matrix)
    
    return v_idx, v_names, a_idx, a_names

# --- UI Sidebar Selection ---
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Select Video Source:", ["Use Default F1 Database", "Upload Custom F1 Video Clip"])
alpha = st.sidebar.slider("Late Fusion Weight (α)", 0.0, 1.0, 0.6, help="Higher defaults to Video, Lower defaults to Audio.")

video_path_to_render = None
active_frames_dir = DEFAULT_FRAMES_DIR

# --- Context Setup Based on Mode ---
if mode == "Use Default F1 Database":
    st.subheader("📁 Searching Default F1 Video Collection")
    st.write("No video needed! Type your query below to automatically search through our pre-indexed race collection.")
    
    try:
        with open("frame_embeddings.pkl", "rb") as f:
            v_data = pickle.load(f)
        with open("audio_embeddings.pkl", "rb") as f:
            a_data = pickle.load(f)
            
        v_index = faiss.IndexFlatIP(v_data["embeddings"].shape[1])
        v_index.add(v_data["embeddings"])
        v_filenames = v_data["filenames"]
        
        a_index = faiss.IndexFlatIP(a_data["embeddings"].shape[1])
        a_index.add(a_data["embeddings"])
        a_filenames = a_data["filenames"]
        
        active_frames_dir = DEFAULT_FRAMES_DIR
    except FileNotFoundError:
        st.warning("⚠️ Default embeddings not found! If deployed, make sure frame_embeddings.pkl and audio_embeddings.pkl are in the repo.")

else:
    st.subheader("📤 Dynamic Video Ingestion Platform")
    uploaded_file = st.file_uploader("Upload an F1 MP4 clip", type=["mp4"])
    if uploaded_file:
        video_path_to_render = os.path.join(UPLOAD_DIR, "uploaded_video.mp4")
        with open(video_path_to_render, "wb") as f:
            f.write(uploaded_file.read())
            
        if "proc_file" not in st.session_state or st.session_state.proc_file != uploaded_file.name:
            with st.spinner("⚡ Processing custom asset... Extracting frames and audio streams..."):
                v_index, v_filenames, a_index, a_filenames = process_custom_video(video_path_to_render)
                st.session_state.v_index, st.session_state.v_filenames = v_index, v_filenames
                st.session_state.a_index, st.session_state.a_filenames = a_index, a_filenames
                st.session_state.proc_file = uploaded_file.name
        else:
            v_index = st.session_state.v_index
            v_filenames = st.session_state.v_filenames
            a_index = st.session_state.a_index
            a_filenames = st.session_state.a_filenames
            
        active_frames_dir = CUSTOM_FRAMES_DIR

# --- Core Execution Search Engine ---
query = st.text_input("🔍 Search for an event:", placeholder="e.g., 'pit stop tire changes' or 'cars screeching and crashing'")

if st.button("Execute Intelligence Query") and query:
    if 'v_index' in locals() or 'v_index' in st.session_state:
        clip_model, clip_processor, clap_model, clap_processor = load_encoders()
        
        # Vectorize text queries
        v_inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            v_out = clip_model.get_text_features(**v_inputs)
        v_vec = v_out.pooler_output if hasattr(v_out, "pooler_output") else (v_out[0] if not isinstance(v_out, torch.Tensor) else v_out)
        v_vec = (v_vec / v_vec.norm(p=2, dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
        
        a_inputs = clap_processor(text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            a_out = clap_model.get_text_features(**a_inputs)
        a_vec = a_out.pooler_output if hasattr(a_out, "pooler_output") else (a_out[0] if not isinstance(a_out, torch.Tensor) else a_out)
        a_vec = (a_vec / a_vec.norm(p=2, dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
        
        # Metric Retrieval mapping
        v_dists, v_idx = v_index.search(v_vec, min(30, len(v_filenames)))
        a_dists, a_idx = a_index.search(a_vec, min(30, len(a_filenames)))
        
        score_map = {}
        for dist, idx in zip(v_dists[0], v_idx[0]):
            base = v_filenames[idx].replace(".jpg", "")
            score_map[base] = score_map.get(base, 0) + (alpha * float(dist))
            
        for dist, idx in zip(a_dists[0], a_idx[0]):
            base = a_filenames[idx].replace(".wav", "")
            score_map[base] = score_map.get(base, 0) + ((1 - alpha) * float(dist))
            
        best_filename = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[0][0]
        
        # Parsing filename maps to isolate timeline positions
        parts = best_filename.split("_frame_")
        vid_prefix = parts[0]
        time_sec = int(parts[1])
        
        # Route to appropriate source file
        if mode == "Use Default F1 Database":
            # For deployment, ensure the MP4s are in the same folder as app_pro.py
            video_path_to_render = f"{vid_prefix.replace('v1_', 'f1_').replace('v2_', 'f1_').replace('v3_', 'f1_').replace('v4_', 'f1_')}.mp4"
            if "pit" in vid_prefix: video_path_to_render = "f1_sample.mp4"
            
        st.success(f"🎯 Event located at {time_sec} seconds in video stream [{vid_prefix}]!")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            try:
                st.video(video_path_to_render, start_time=time_sec)
            except Exception:
                st.warning(f"Video file {video_path_to_render} not found. Ensure it is uploaded to your GitHub repo!")
            
        with col2:
            st.subheader("🎙️ Live Broadcast AI Commentary")
            # Determine the correct frame path
            if mode != "Use Default F1 Database":
                frame_path = os.path.join(active_frames_dir, f"{best_filename}.jpg")
            else:
                frame_path = os.path.join(active_frames_dir, f"{vid_prefix}_frame_{time_sec:04d}.jpg")
            
            if os.path.exists(frame_path):
                with st.spinner("🤖 Observer Agent translating visual data..."):
                    clinical_json = extract_clinical_json_via_api(frame_path)
                
                with st.spinner("🎙️ Generating Commentary..."):
                    commentary = generate_croft_commentary(clinical_json)
                
                st.info(f"**{commentary}**")
            else:
                st.warning("Frame image not found for commentary generation.")
    else:
        st.error("Please load or select a database mode before executing query.")

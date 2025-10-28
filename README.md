
# 🎥 (Gemini & Ollama) POC

This project provides Python scripts to analyze a directory of video files for **situational awareness and safety**, focusing primarily on detecting **human presence**.

It supports two analysis backends:

* **🧠 Google Gemini API (Recommended):** Performs **multi-frame, full video analysis** using Google's video models.
* **💻 Ollama (Local):** Performs **single-frame, fast analysis** using a local vision model.

---

## 📜 Script Overview

| Script                     | Backend | Description                                                                        |
| -------------------------- | ------- | ---------------------------------------------------------------------------------- |
| `res2CSV.py`               | Gemini  | Analyzes full videos and outputs results into a single CSV file.                   |
| `res2json.py`              | Gemini  | Analyzes full videos and saves each result as a separate JSON file.                |
| `video_analyzer_ollama.py` | Ollama  | Analyzes the first frame of each video and outputs results into a single CSV file. |

---

## ✨ Features

* 🔍 Recursively scans directories for video files (`.mp4`, `.mov`, `.avi`, `.mkv`)
* 🧾 Enforces a **strict JSON schema** using detailed prompts
* ✅ Validates model output with **Pydantic**
* 📊 Generates a single CSV or multiple JSON outputs
* 🧩 Includes **robust error handling** for incomplete or invalid Ollama responses

---

## ⚙️ Requirements

* **Python 3.7+**
* **Google API Key** (for Gemini)
* **Local Ollama instance** (for local analysis)
* Dependencies listed in `requirements.txt`

---

## 🚀 Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy the example `.env` file and update it with your details:

```bash
copy .env.example .env
```

---

## 🔧 Backend Setup (Choose One or Both)

### ☁️ Gemini (Cloud)

1. Get your **GOOGLE_API_KEY**.
2. Add it to your `.env` file.

### 🖥️ Ollama (Local)

1. Install and run **Ollama** using Docker. Follow official setup instructions:
   👉 [https://github.com/ollama/ollama](https://github.com/ollama/ollama)
2. Pull the required vision model:

   ```bash
   docker exec -it ollama ollama pull llama3.2-vision:11b-instruct-q4_K_M
   ```

---

## 🧩 `.env` Configuration

Example `.env` file:

```bash
# --- Gemini ---
GOOGLE_API_KEY=YOUR_API_KEY_HERE 

# --- Ollama ---
OLLAMA_HOST="http://localhost:11434"

# --- Common ---
ROOT_DIR="./vid"               # Directory with your video files
OUTPUT_DIR="./analysis_results" # Directory to save results
VIDEO_FPS=1                    # Frame sampling rate
```

---

## ▶️ Usage

Ensure `.env` is configured and the chosen backend is ready.

### Gemini – Full Video → CSV

```bash
python res2CSV.py
```

### Gemini – Full Video → JSON Files

```bash
python res2json.py
```

### Ollama – First Frame → CSV

```bash
python video_analyzer_ollama.py
```

---

## 📑 Output CSV Columns

| Column                                                                | Description                     |
| --------------------------------------------------------------------- | ------------------------------- |
| `video_name`                                                          | Video file name                 |
| `persons_value`, `persons_confidence`                                 | Human presence detection        |
| `night_day_unknown_value`, `night_day_unknown_confidence`             | Lighting condition              |
| `tank_movement_value`, `tank_movement_confidence`                     | Tank movement detection         |
| `turret_movement_value`, `turret_movement_confidence`                 | Turret movement                 |
| `area_type_value`, `area_type_confidence`                             | Scene type classification       |
| `distance_category_value`, `distance_category_confidence`             | Distance estimation             |
| `faces_visible_value`, `faces_visible_confidence`                     | Face visibility                 |
| `tokens_prompt`, `tokens_thinking`, `tokens_response`, `tokens_total` | Token usage metrics             |
| `video_path`                                                          | Full path to the analyzed video |

> ⚠️ *For Ollama runs, if the model output is invalid or incomplete, `_value` and `_confidence` fields may show `N/A`.*

---

## 🧠 Author

Developed by **Rei Naftali**
---

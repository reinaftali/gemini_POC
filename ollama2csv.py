import os
import json
import logging
import csv
import base64
from pathlib import Path
from typing import List, TypeVar, Generic, Optional
from enum import Enum
from dotenv import load_dotenv

import ollama
import cv2
from pydantic import BaseModel, Field

# --- Setup Logging and Environment ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Globals ---
ROOT_DIR = Path(os.getenv("ROOT_DIR"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", 1))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = "llama3.2-vision:11b-instruct-q4_K_M"

OUTPUT_DIR.mkdir(exist_ok=True)

# --- MODIFIED PROMPT ---
PROMPT = """
You are an expert safety AI analyst. Your mission is to analyze a single image for immediate situational awareness (safety and reconnaissance). Accuracy and self-assessment of that accuracy are paramount.

Your **number one priority** is to detect human presence. A false negative (missing a person) is a critical failure. All other tasks are secondary.

---
**CRITICAL RULES**
---
1.  **Return ONLY a valid JSON object.** Do not include explanations, apologies, or any surrounding text.
2.  **Prioritize Detection Over Confidence:** If you see any shape, movement, or object in any frame that could possibly be a person or human body part, you **MUST** set `"persons": {"value": "Yes"}`. It is better to report "Yes" with low confidence than to miss a person.
3.  **Provide Confidence:** For each assessment, provide a `value` and a `confidence` score from 0.0 (low) to 1.0 (high).
4.  **Single Frame Analysis:** You are analyzing only **one single image**. You **CANNOT** detect movement. Therefore, you **MUST** set both `"tank_movement"` and `"turret_movement"` to `{"value": "No", "confidence": 1.0}`. This is mandatory.
5.  **Assess All Fields:** You **MUST** provide a value for all fields in the schema, even if your confidence is low.

---
**EXAMPLES**
---

**Example 1: Clear person, close range, stationary vehicle.**
* Image Description: A single person is clearly visible walking about 10 meters away from the vehicle.
* Output:
{
    "persons": {"value": "Yes", "confidence": 0.99},
    "night_day_unknown": {"value": "Day", "confidence": 1.0},
    "tank_movement": {"value": "No", "confidence": 1.0},
    "turret_movement": {"value": "No", "confidence": 1.0},
    "area_type": {"value": "Urban", "confidence": 0.95},
    "distance_category": {"value": 2, "confidence": 0.9},
    "faces_visible": {"value": "No", "confidence": 0.9}
}"""


# --- Pydantic Schema (Unchanged Enums) ---
class YesNo(str, Enum):
    YES = "Yes"
    NO = "No"

class DayNight(str, Enum):
    DAY = "Day"
    NIGHT = "Night"
    UNKNOWN = "Unknown"

class AreaType(str,Enum):
    URBAN = "Urban"
    OPEN = "Open"
    SHELTER = "Shelter"

T = TypeVar('T')

class ConfidenceItem(BaseModel, Generic[T]):
    """A generic model to hold a value and its confidence score."""
    value: T
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The AI's confidence in the value, from 0.0 (no confidence) to 1.0 (certainty)."
    )

# --- MODIFIED VideoAnalysis Schema ---
class VideoAnalysis(BaseModel):
    """
    Pydantic schema for video analysis results.
    All fields are optional to handle inconsistent model output.
    """
    video_path: str = ""
    persons: Optional[ConfidenceItem[YesNo]] = None
    night_day_unknown: Optional[ConfidenceItem[DayNight]] = None
    tank_movement: Optional[ConfidenceItem[YesNo]] = None
    turret_movement: Optional[ConfidenceItem[YesNo]] = None
    area_type: Optional[ConfidenceItem[AreaType]] = None
    distance_category: Optional[ConfidenceItem[int]] = Field(
        default=None,
        description=(
            "Integer from 0-4 representing the distance to the CLOSEST person. "
            "0: No Person. 1: Contact (0-2m). 2: Close (2-15m). "
            "3: Mid (15-50m). 4: Far (50m+)."
        )
    )
    faces_visible: Optional[ConfidenceItem[YesNo]] = None
    tokens_prompt: int = 0
    tokens_thinking: int = 0  # Note: This will be 0 for Ollama
    tokens_response: int = 0
    tokens_total: int = 0


# --- Frame Extraction (Unchanged) ---
def extract_frames(video_path: Path, target_fps: int) -> List[str]:
    """Extracts frames from a video at a target FPS and returns them as base64 strings."""
    frames = []
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Cannot open video file: {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0:
            logging.warning(f"Video FPS reported as 0 for {video_path.name}, defaulting to 30.")
            video_fps = 30 # Set a reasonable default

        frame_skip = int(round(video_fps / target_fps))
        if frame_skip <= 0:
            frame_skip = 1

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                success, buffer = cv2.imencode('.jpg', frame)
                if not success:
                    logging.warning(f"Failed to encode frame {frame_count} from {video_path.name}")
                    continue
                
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames.append(frame_base64)
            
            frame_count += 1

        cap.release()
        logging.info(f"Extracted {len(frames)} frames from {video_path.name} (target FPS: {target_fps})")
        return frames

    except Exception as e:
        logging.error(f"Error extracting frames from {video_path.name}: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return []

# --- Video Listing (Unchanged) ---
def list_videos(root_dir: Path) -> List[Path]:
    """Recursively finds all video files in a directory."""
    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    return [
        p for p in root_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in video_extensions
    ]

# --- MODIFIED process_video Function ---
def process_video(video_path: Path, client: ollama.Client) -> Optional[VideoAnalysis]:
    """Analyzes a single video and returns the structured analysis data."""
    logging.info(f"Processing: {video_path.name}")
    
    frames_base64 = extract_frames(video_path, VIDEO_FPS)
    if not frames_base64:
        logging.error(f"No frames extracted for {video_path.name}. Skipping.")
        return None

    try:
        # Prepare content - send only the FIRST frame
        messages = [
            {
                'role': 'user',
                'content': PROMPT,
                'images': [frames_base64[0]] # Send only the first frame
            }
        ]

        logging.info(f"Sending 1 (of {len(frames_base64)} extracted) frame to single-image model {MODEL_NAME}...")
        response = client.chat(
            model=MODEL_NAME,
            messages=messages,
            format='json'
        )
        
        # Parse the JSON response string
        try:
            response_string = response['message']['content']
            # Clean up potential markdown formatting
            if response_string.startswith("```json"):
                response_string = response_string[7:]
            if response_string.endswith("```"):
                response_string = response_string[:-3]
            
            response_json = json.loads(response_string.strip())
            
            # Use parse_obj to handle potential missing fields gracefully
            result = VideoAnalysis.model_validate(response_json)

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response from Ollama for {video_path.name}: {e}")
            logging.error(f"Raw response: {response_string}")
            return None
        except Exception as pydantic_error:
            # This error will now catch validation errors if the *type* is wrong,
            # but not if a field is missing (since they are Optional).
            logging.error(f"Pydantic validation error for {video_path.name}: {pydantic_error}")
            logging.error(f"Raw JSON: {response_json}")
            return None

        # Finalize the analysis object
        result.video_path = str(video_path.relative_to(ROOT_DIR))
        result.tokens_prompt = response.get('prompt_eval_count', 0)
        result.tokens_thinking = 0
        result.tokens_response = response.get('eval_count', 0)
        result.tokens_total = result.tokens_prompt + result.tokens_response
        
        logging.info(f"Successfully analyzed: {video_path.name}")
        logging.info(
            f"  Tokens -> prompt: {result.tokens_prompt}, "
            f"response: {result.tokens_response}, total: {result.tokens_total}"
        )
        return result

    except ollama.ResponseError as e:
        logging.error(f"Ollama API error for {video_path.name}: {e.error}")
        return None
    except Exception as e:
        logging.error(f"Failed to analyze {video_path.name}: {e}")
        return None


# --- MODIFIED main Function ---
def main():
    """Initialize the client, process all videos, and save results to a single CSV."""
    if not ROOT_DIR.is_dir():
        logging.error(f"Video root directory not found: {ROOT_DIR}")
        return

    try:
        client = ollama.Client(host=OLLAMA_HOST)
        client.list()
        logging.info(f"Connected to Ollama at {OLLAMA_HOST}")
    except Exception as e:
        logging.error(f"Failed to connect to Ollama at {OLLAMA_HOST}. Is it running?")
        logging.error(e)
        return

    videos = list_videos(ROOT_DIR)

    if not videos:
        logging.warning(f"No video files found in {ROOT_DIR}")
        return

    logging.info(f"Found {len(videos)} video(s) under {ROOT_DIR}")

    csv_filename = f"{MODEL_NAME.replace(':', '_')}_{VIDEO_FPS}fps.csv"
    csv_output_path = OUTPUT_DIR / csv_filename
    
    header = [
        "video_name",
        "persons_value", "persons_confidence",
        "night_day_unknown_value", "night_day_unknown_confidence",
        "tank_movement_value", "tank_movement_confidence",
        "turret_movement_value", "turret_movement_confidence",
        "area_type_value", "area_type_confidence",
        "distance_category_value", "distance_category_confidence",
        "faces_visible_value", "faces_visible_confidence",
        "tokens_prompt", "tokens_thinking", "tokens_response", "tokens_total",
        "video_path"
    ]

    try:
        with open(csv_output_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            logging.info(f"Writing results to: {csv_output_path}")

            for idx, video_path in enumerate(videos, start=1):
                logging.info(f"\n--- Video {idx}/{len(videos)} ---")
                result = process_video(video_path, client)

                # --- MODIFIED CSV ROW WRITING ---
                if result:
                    # Helper functions to safely get values or 'N/A'
                    def get_val(item: Optional[ConfidenceItem]):
                        return item.value if item else 'N/A'
                    
                    def get_conf(item: Optional[ConfidenceItem]):
                        return item.confidence if item else 'N/A'

                    row = [
                        video_path.stem,  # video_name
                        get_val(result.persons), get_conf(result.persons),
                        get_val(result.night_day_unknown), get_conf(result.night_day_unknown),
                        get_val(result.tank_movement), get_conf(result.tank_movement),
                        get_val(result.turret_movement), get_conf(result.turret_movement),
                        get_val(result.area_type), get_conf(result.area_type),
                        get_val(result.distance_category), get_conf(result.distance_category),
                        get_val(result.faces_visible), get_conf(result.faces_visible),
                        result.tokens_prompt, result.tokens_thinking,
                        result.tokens_response, result.tokens_total,
                        result.video_path
                    ]
                    writer.writerow(row)
                else:
                    logging.warning(f"Skipping CSV entry for failed video: {video_path.name}")
        
        logging.info(f"\nAll videos processed. Results saved in: {csv_output_ch}")

    except IOError as e:
        logging.error(f"Failed to write to CSV file {csv_output_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during CSV writing: {e}")


if __name__ == "__main__":
    main()
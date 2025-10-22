import os
import json
import logging
import csv  
from pathlib import Path
from typing import List, TypeVar, Generic, Optional  
from enum import Enum
from dotenv import load_dotenv

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

ROOT_DIR = Path(os.getenv("ROOT_DIR"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS"))
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-pro"


OUTPUT_DIR.mkdir(exist_ok=True)

PROMPT = """
You are an expert safety AI analyst. Your mission is to analyze video for immediate situational awareness (safety and reconnaissance). Accuracy and self-assessment of that accuracy are paramount.

Your **number one priority** is to detect human presence. A false negative (missing a person) is a critical failure. All other tasks are secondary.

---
**CRITICAL RULES**
---
1.  **Return ONLY a valid JSON object.** Do not include explanations, apologies, or any surrounding text like ```json.
2.  **Prioritize Detection Over Confidence:** If you see any shape, movement, or object that could possibly be a person or human body part, you **MUST** set `"persons": {"value": "Yes"}`. It is better to report "Yes" with low confidence than to miss a person.
3.  **Assess the Entire Video:** Scan the entire duration of the video for your analysis.
4.  **Provide Confidence:** For each assessment, provide a `value` and a `confidence` score from 0.0 (low) to 1.0 (high).
5.  **Distinguish Movement:** Pay close attention to the difference between the tank's hull moving (`tank_movement`) and its turret rotating (`turret_movement`).

---
**EXAMPLES**
---

**Example 1: Clear person, close range, stationary vehicle.**
* Video Description: A single person is clearly visible walking about 10 meters away from the vehicle. The camera is not moving.
* Output:
```json
{
    "persons": {"value": "Yes", "confidence": 0.99},
    "night_day_unknown": {"value": "Day", "confidence": 1.0},
    "tank_movement": {"value": "No", "confidence": 1.0},
    "turret_movement": {"value": "No", "confidence": 1.0},
    "area_type": {"value": "Urban", "confidence": 0.95},
    "distance_category": {"value": 2, "confidence": 0.9},
    "faces_visible": {"value": "No", "confidence": 0.9}
}"""


#SCHEMA
class YesNo(str, Enum):
    YES = "Yes"
    NO = "No"

class DayNight(str, Enum):
    DAY = "Day"
    NIGHT = "Night"
    UNKNOWN = "Unknown"

class AreaType(str, Enum):
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

class VideoAnalysis(BaseModel):
    """Pydantic schema for video analysis results."""
    video_path: str = ""
    persons: ConfidenceItem[YesNo]
    night_day_unknown: ConfidenceItem[DayNight]
    tank_movement: ConfidenceItem[YesNo]
    turret_movement: ConfidenceItem[YesNo]
    area_type: ConfidenceItem[AreaType]
    distance_category: ConfidenceItem[int] = Field(
        ...,
        description=(
            "Integer from 0-4 representing the distance to the CLOSEST person. "
            "0: No Person. 1: Contact (0-2m). 2: Close (2-15m). "
            "3: Mid (15-50m). 4: Far (50m+)."
        )
    )
    faces_visible: ConfidenceItem[YesNo]
    tokens_prompt: int = 0
    tokens_thinking: int = 0
    tokens_response: int = 0
    tokens_total: int = 0

def list_videos(root_dir: Path) -> List[Path]:
    """Recursively finds all video files in a directory."""
    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    return [
        p for p in root_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in video_extensions
    ]

def process_video(video_path: Path, client: genai.Client) -> Optional[VideoAnalysis]:
    """Analyzes a single video and returns the structured analysis data."""
    logging.info(f"Processing: {video_path.name}")
    try:
        #Prepare content
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        content = types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
                    video_metadata=types.VideoMetadata(fps=VIDEO_FPS)
                ),
                types.Part(text=PROMPT)
            ]
        )
        prompt_part = types.Content(parts=[types.Part(text=PROMPT)])

        # Count tokens
        total_tokens_info = client.models.count_tokens(model=MODEL_NAME, contents=content)
        prompt_tokens_info = client.models.count_tokens(model=MODEL_NAME, contents=prompt_part)
        
        tokens_prompt = prompt_tokens_info.total_tokens
        tokens_thinking = total_tokens_info.total_tokens - tokens_prompt

        # Generate structured response
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=content,
            config={
                "response_mime_type": "application/json",
                "response_schema": VideoAnalysis
            }
        )
        
        #Finalize the analysis object
        result = response.parsed
        result.video_path = str(video_path.relative_to(ROOT_DIR))
        result.tokens_prompt = tokens_prompt
        result.tokens_thinking = tokens_thinking
        result.tokens_response = getattr(response.usage_metadata, "candidates_token_count", 0)
        result.tokens_total = result.tokens_prompt + result.tokens_thinking + result.tokens_response
        
        # --- MODIFICATION ---
        # Instead of saving to JSON, log success and return the result object
        logging.info(f"Successfully analyzed: {video_path.name}")
        logging.info(
            f"  Tokens -> prompt: {result.tokens_prompt}, thinking: {result.tokens_thinking}, "
            f"response: {result.tokens_response}, total: {result.tokens_total}"
        )
        return result

    except Exception as e:
        logging.error(f"Failed to analyze {video_path.name}: {e}")
        return None  # Return None on failure


def main():
    """Initialize the client, process all videos, and save results to a single CSV."""
    if not API_KEY:
        logging.error("GOOGLE_API_KEY environment variable not set.")
        return

    if not ROOT_DIR.is_dir():
        logging.error(f"Video root directory not found: {ROOT_DIR}")
        return

    client = genai.Client(api_key=API_KEY)
    videos = list_videos(ROOT_DIR)

    if not videos:
        logging.warning(f"No video files found in {ROOT_DIR}")
        return

    logging.info(f"Found {len(videos)} video(s) under {ROOT_DIR}")

    # --- NEW CSV WRITING LOGIC ---
    csv_filename = f"{MODEL_NAME}_{VIDEO_FPS}fps.csv"
    csv_output_path = OUTPUT_DIR / csv_filename
    
    # Define header row based on your requirements
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
        # Open the CSV file once in write mode
        with open(csv_output_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)  # Write the header row
            
            logging.info(f"Writing results to: {csv_output_path}")

            # Process each video
            for idx, video_path in enumerate(videos, start=1):
                logging.info(f"\n--- Video {idx}/{len(videos)} ---")
                result = process_video(video_path, client)  # Get result or None

                # If analysis was successful, write the row
                if result:
                    # Flatten the result object into a list matching the header
                    row = [
                        video_path.stem,  # video_name (as requested)
                        result.persons.value, result.persons.confidence,
                        result.night_day_unknown.value, result.night_day_unknown.confidence,
                        result.tank_movement.value, result.tank_movement.confidence,
                        result.turret_movement.value, result.turret_movement.confidence,
                        result.area_type.value, result.area_type.confidence,
                        result.distance_category.value, result.distance_category.confidence,
                        result.faces_visible.value, result.faces_visible.confidence,
                        result.tokens_prompt, result.tokens_thinking,
                        result.tokens_response, result.tokens_total,
                        result.video_path
                    ]
                    writer.writerow(row)
                else:
                    logging.warning(f"Skipping CSV entry for failed video: {video_path.name}")
        
        logging.info(f"\nAll videos processed. Results saved in: {csv_output_path}")

    except IOError as e:
        logging.error(f"Failed to write to CSV file {csv_output_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during CSV writing: {e}")
    # --- END OF NEW CSV LOGIC ---

if __name__ == "__main__":
    main()
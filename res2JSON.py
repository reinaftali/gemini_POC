import os
import json
import logging
from pathlib import Path
from typing import List
from enum import Enum
from dotenv import load_dotenv

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import TypeVar, Generic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

ROOT_DIR = Path(os.getenv("ROOT_DIR"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS"))
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash"


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

def process_video(video_path: Path, client: genai.Client) -> None:
    """Analyzes a single video, retrieves structured data, and saves it to a JSON file."""
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
        
        #Save result
        output_filename = video_path.stem + ".json"
        output_path = OUTPUT_DIR / output_filename
        with open(output_path, "w") as f:
            f.write(result.model_dump_json(indent=2))

        logging.info(f"Saved result -> {output_path}")
        logging.info(
            f"  Tokens -> prompt: {result.tokens_prompt}, thinking: {result.tokens_thinking}, "
            f"response: {result.tokens_response}, total: {result.tokens_total}"
        )

    except Exception as e:
        logging.error(f"Failed to analyze {video_path.name}: {e}")


def main():
    """initialize the client and process all videos."""
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

    for idx, video_path in enumerate(videos, start=1):
        logging.info(f"\n--- Video {idx}/{len(videos)} ---")
        process_video(video_path, client)

    logging.info(f"\nAll videos processed. Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
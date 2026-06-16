import json
from typing import Any, Dict, List
import pandas as pd

def model_to_json_str(model_data: Any) -> str:
    """
    Converts a Pydantic model, dictionary, or primitive object to a formatted JSON string.
    
    Args:
        model_data: Pydantic model instance or dict.
        
    Returns:
        str: Pretty-printed JSON string.
    """
    if hasattr(model_data, "model_dump"):
        # Pydantic v2
        return json.dumps(model_data.model_dump(), indent=2)
    elif hasattr(model_data, "dict"):
        # Pydantic v1 fallback
        return json.dumps(model_data.dict(), indent=2)
    return json.dumps(model_data, indent=2)

def quiz_attempts_to_df(attempts: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts a list of quiz attempts into a structured Pandas DataFrame.
    
    Args:
        attempts: List of quiz attempt dictionaries.
        
    Returns:
        pd.DataFrame: Pandas DataFrame with formatted quiz reports.
    """
    if not attempts:
        return pd.DataFrame(
            columns=["id", "topic", "timestamp", "score", "total_questions", "percentage"]
        )
    df = pd.DataFrame(attempts)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def get_difficulty_color(difficulty: str) -> str:
    """
    Returns a hex color associated with a difficulty level.
    """
    diff_lower = difficulty.lower()
    if "easy" in diff_lower or "beginner" in diff_lower:
        return "#00FF66" # Bright Green
    elif "medium" in diff_lower or "intermediate" in diff_lower:
        return "#FFCC00" # Orange/Yellow
    elif "hard" in diff_lower or "advanced" in diff_lower:
        return "#FF3366" # Red/Pink
    return "#00F0FF" # Cyan default

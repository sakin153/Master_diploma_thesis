import json

import numpy as np


def parse_output_to_json(output):
    """Parse LLM output to JSON with robust error handling."""
    if "```json\n" in output:
        json_start = output.find("```json\n") + len("```json\n")
        json_end = output.find("\n```", json_start)
        json_str = output[json_start:json_end].strip()
    else:
        json_str = output.strip()

    # Clean up common LLM artifacts that break JSON parsing
    import re
    
    # CRITICAL: Remove ALL non-ASCII characters (Chinese, Russian, etc.)
    # This must be done BEFORE JSON parsing to prevent syntax errors
    json_str = re.sub(r'[^\x00-\x7F]+', '', json_str)
    
    # Remove any remaining problematic characters
    json_str = json_str.replace('极', '')
    
    # Fix common JSON formatting issues
    # Remove any text between closing brace and opening brace (comments/descriptions)
    json_str = re.sub(r'\}\s*[^,\[\]\{\}]+\s*\{', '},{', json_str)
    
    try:
        parsed_json = json.loads(json_str)
    except json.decoder.JSONDecodeError as e:
        # Log the error with context
        import logging
        logging.error(f"JSON parsing failed: {e}")
        logging.error(f"Problematic JSON (first 500 chars): {json_str[:500]}")
        raise RuntimeError(f"Error parsing output. Error - {e}")

    return parsed_json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

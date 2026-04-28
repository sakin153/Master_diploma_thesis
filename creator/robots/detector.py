"""Robot detection in user prompts."""
import re
from typing import List
from creator.robots.catalog import load_robot_catalog, find_robot_by_keyword


def detect_robots(query: str) -> List[str]:
    """Detect robot mentions in user query.
    
    Returns list of robot IDs (e.g., ["franka_fr3", "unitree_a1"]).
    """
    query_lower = query.lower()
    catalog = load_robot_catalog()
    detected = []
    
    for robot_id, info in catalog.items():
        keywords = info.get("keywords", [])
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Skip generic terms that could match multiple robots
            if keyword_lower in ["manipulator", "манипулятор", "robot", "робот"]:
                continue
            
            # Normalize separators: _ and - become space for matching
            normalized_query = query_lower.replace('_', ' ').replace('-', ' ')
            normalized_keyword = keyword_lower.replace('_', ' ').replace('-', ' ')
            
            # Simple substring match
            if normalized_keyword in normalized_query:
                if robot_id not in detected:
                    detected.append(robot_id)
                break
    
    return detected

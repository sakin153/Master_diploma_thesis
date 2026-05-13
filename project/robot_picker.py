"""Stage 0.5 - Robot Detection and Selection
Detects robots in user query via LLM and matches them to robot catalog
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from project.llm_request import DEFAULT_MODEL, request as _default_request


def _load_prompt(filename):
    """Load a prompt from the prompts directory."""
    prompt_file = Path(__file__).parent / "prompts" / filename
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


_ROBOT_DETECTION_PROMPT = _load_prompt("robot_detection.txt")


def _tokenize(text):
    """Tokenize text for keyword matching (supports Latin and Cyrillic)."""
    return re.findall(r"[a-zа-яё0-9]+", (text or "").lower())


def _score_robot(mention: str, robot_id: str, robot_info: Dict) -> int:
    """Score how well a robot matches the user's mention.
    
    Similar to model_picker._score() but adapted for robots.
    """
    mention_lower = mention.lower()
    robot_name = str(robot_info.get("name", "")).lower()
    keywords = robot_info.get("keywords", [])
    robot_type = str(robot_info.get("type", "")).lower()
    
    score = 0
    
    # Exact match with robot_id or name
    if mention_lower == robot_id.lower():
        score += 25
    elif mention_lower == robot_name:
        score += 20
    
    # Partial match in name
    if mention_lower in robot_name or robot_name in mention_lower:
        score += 12
    
    # Keyword matching
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if mention_lower == keyword_lower:
            score += 15
        elif keyword_lower in mention_lower or mention_lower in keyword_lower:
            score += 8
    
    # Token-based matching
    mention_tokens = set(_tokenize(mention_lower))
    name_tokens = set(_tokenize(robot_name))
    keyword_tokens = set()
    for kw in keywords:
        keyword_tokens.update(_tokenize(kw))
    
    # Bonus for matching tokens
    common_name = mention_tokens & name_tokens
    common_keywords = mention_tokens & keyword_tokens
    score += len(common_name) * 5
    score += len(common_keywords) * 3
    
    # Type matching bonus
    if robot_type in mention_lower:
        score += 6
    
    return score


def _rank_robots(mention: str, robot_catalog: Dict, limit: int = 5) -> List[tuple]:
    """Rank robots by relevance to the mention.
    
    Returns list of (robot_id, robot_info, score) tuples.
    """
    scored = []
    for robot_id, robot_info in robot_catalog.items():
        score = _score_robot(mention, robot_id, robot_info)
        if score >= 5:  # Minimum threshold
            scored.append((robot_id, robot_info, score))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:limit]


def _llm_disambiguate_robot(mention: str, candidates: List[tuple], query: str, llm) -> Optional[str]:
    """Use LLM to choose the best robot when multiple candidates exist.
    
    Args:
        mention: The robot mention from user query
        candidates: List of (robot_id, robot_info, score) tuples
        query: Original user query for context
        llm: LLM request function
        
    Returns:
        robot_id of chosen robot, or None if no good match
    """
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0][0]
    
    # Prepare candidate list for LLM
    candidate_list = []
    for robot_id, robot_info, score in candidates:
        candidate_list.append({
            "robot_id": robot_id,
            "name": robot_info.get("name", ""),
            "type": robot_info.get("type", ""),
            "description": robot_info.get("description", ""),
            "keywords": robot_info.get("keywords", [])
        })
    
    prompt = f"""You are a robot selection assistant. The user mentioned "{mention}" in their scene description.

User's full query: "{query}"

Available robots:
{json.dumps(candidate_list, ensure_ascii=False, indent=2)}

Choose the BEST matching robot for the user's intent. Consider:
1. Name similarity
2. Type appropriateness
3. Context from the full query

Return ONLY a JSON object with the robot_id:
{{"robot_id": "chosen_robot_id"}}

If none match well, return:
{{"robot_id": "none"}}
"""
    
    try:
        result = llm(prompt, mention)
    except Exception as e:
        print(f"[robot_picker] LLM error for '{mention}': {e}")
        # Fallback to highest scored
        return candidates[0][0]
    
    # Parse LLM response
    chosen_id = ""
    if isinstance(result, dict):
        chosen_id = str(result.get("robot_id", "")).strip()
    elif isinstance(result, str):
        # Try to extract robot_id from text
        match = re.search(r'"robot_id"\s*:\s*"([^"]+)"', result)
        if match:
            chosen_id = match.group(1)
    
    if chosen_id.lower() == "none" or not chosen_id:
        return None
    
    # Verify chosen_id exists in candidates
    candidate_ids = {rid for rid, _, _ in candidates}
    if chosen_id in candidate_ids:
        return chosen_id
    
    # Fallback to highest scored
    return candidates[0][0]


def detect_robots_in_query(query: str, llm) -> List[Dict]:
    """Stage 0.5a: Detect robot mentions in user query via LLM.
    
    Args:
        query: User's scene description
        llm: LLM request function
        
    Returns:
        List of robot mentions with context:
        [
            {
                "mentioned_as": "franka robot",
                "robot_type": "manipulator",
                "placement_hint": "table_mounted",
                "task_context": "pick and place"
            },
            ...
        ]
    """
    prompt = _ROBOT_DETECTION_PROMPT.format(query=query)
    
    try:
        result = llm(prompt, "robot_detection")
    except Exception as e:
        print(f"[robot_picker] LLM detection error: {e}")
        return []
    
    # Parse LLM response
    if isinstance(result, dict):
        data = result
    elif isinstance(result, str):
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            print(f"[robot_picker] Failed to parse LLM response as JSON")
            return []
    else:
        return []
    
    if not data.get("robots_detected", False):
        return []
    
    return data.get("robots", [])


def pick_robots(query: str, robot_catalog: Dict, llm_model: str = DEFAULT_MODEL, 
                prompt_model_fn=None) -> List[Dict]:
    """Main function: Detect and select robots from catalog.
    
    Pipeline:
        1. LLM detects robot mentions in query
        2. For each mention, rank catalog robots by keyword/name matching
        3. If multiple candidates, use LLM to disambiguate
        4. Return list of selected robots with metadata
    
    Args:
        query: User's scene description
        robot_catalog: Dict of robot_id -> robot_info
        llm_model: LLM model name
        prompt_model_fn: Optional custom LLM function
        
    Returns:
        List of selected robots:
        [
            {
                "robot_id": "franka_fr3",
                "robot_info": {...},
                "mentioned_as": "franka robot",
                "placement_hint": "table_mounted",
                "task_context": "pick and place"
            },
            ...
        ]
    """
    llm = prompt_model_fn if prompt_model_fn is not None else _default_request
    
    # Stage 1: Detect robots in query
    detected = detect_robots_in_query(query, llm)
    
    if not detected:
        print("[robot_picker] No robots detected in query")
        return []
    
    print(f"[robot_picker] Detected {len(detected)} robot mention(s)")
    
    selected_robots = []
    
    # Stage 2: Match each mention to catalog
    for mention_data in detected:
        mention = mention_data.get("mentioned_as", "")
        if not mention:
            continue
        
        print(f"[robot_picker] Processing mention: '{mention}'")
        
        # Rank robots by relevance
        candidates = _rank_robots(mention, robot_catalog, limit=5)
        
        if not candidates:
            print(f"[robot_picker] No matching robots found for '{mention}'")
            continue
        
        print(f"[robot_picker] Found {len(candidates)} candidate(s)")
        
        # Disambiguate if needed
        chosen_id = _llm_disambiguate_robot(mention, candidates, query, llm)
        
        if not chosen_id:
            print(f"[robot_picker] LLM rejected all candidates for '{mention}'")
            continue
        
        # Get full robot info
        robot_info = robot_catalog.get(chosen_id)
        if not robot_info:
            continue
        
        print(f"[robot_picker] Selected: {chosen_id} ({robot_info.get('name', 'unknown')})")
        
        # Combine detection data with catalog data
        selected_robots.append({
            "robot_id": chosen_id,
            "robot_info": robot_info,
            "mentioned_as": mention,
            "robot_type": mention_data.get("robot_type", "other"),
            "placement_hint": mention_data.get("placement_hint", "unspecified"),
            "task_context": mention_data.get("task_context", "")
        })
    
    return selected_robots

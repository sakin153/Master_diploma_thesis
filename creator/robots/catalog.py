"""Robot catalog loader."""
import json
from pathlib import Path
from typing import Dict, List, Optional


_CATALOG_PATH = Path(__file__).parent.parent.parent / "robot_assets" / "robots.json"
_CATALOG_CACHE: Optional[Dict] = None


def load_robot_catalog() -> Dict:
    """Load robot catalog from robots.json."""
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE
    
    with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
        _CATALOG_CACHE = json.load(f)
    return _CATALOG_CACHE


def get_robot_info(robot_id: str) -> Optional[Dict]:
    """Get robot metadata by ID."""
    catalog = load_robot_catalog()
    return catalog.get(robot_id)


def list_robots() -> List[str]:
    """List all available robot IDs."""
    catalog = load_robot_catalog()
    return list(catalog.keys())


def find_robot_by_keyword(keyword: str) -> Optional[str]:
    """Find robot ID by keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    catalog = load_robot_catalog()
    
    for robot_id, info in catalog.items():
        keywords = info.get("keywords", [])
        if any(kw.lower() == keyword_lower or keyword_lower in kw.lower() for kw in keywords):
            return robot_id
    return None

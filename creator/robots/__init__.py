"""Robot integration module."""
from creator.robots.catalog import load_robot_catalog, get_robot_info, list_robots
from creator.robots.detector import detect_robots
from creator.robots.placer import place_robots

__all__ = [
    "load_robot_catalog",
    "get_robot_info",
    "list_robots",
    "detect_robots",
    "place_robots",
]

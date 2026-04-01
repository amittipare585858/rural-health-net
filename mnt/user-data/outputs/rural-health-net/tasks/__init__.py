from tasks.easy   import EASY_CONFIG
from tasks.medium import MEDIUM_CONFIG
from tasks.hard   import HARD_CONFIG

TASK_CONFIGS = {
    "easy":   EASY_CONFIG,
    "medium": MEDIUM_CONFIG,
    "hard":   HARD_CONFIG,
}

__all__ = ["TASK_CONFIGS", "EASY_CONFIG", "MEDIUM_CONFIG", "HARD_CONFIG"]

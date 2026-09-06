from typing import TypedDict, Annotated
from pydantic import BaseModel
import operator

# The Core Metrics Schema
class PlayerMetrics(BaseModel):
    tech: int = 1
    charm: int = 1
    fitness: int = 1
    language: int = 1
    stress_tolerance: int = 1
    presence: int = 1
    wealth: int = 1
    intellect: int = 1
    alignment: int = 5       # Scale of 1 (Ruthless) to 10 (Compassionate)
    combat: int = 1
    cautiousness: int = 1

# The Global Graph State
class EngineState(TypedDict):
    """
    This is the global dictionary that LangGraph will pass from node to node.
    Every node reads this, updates a piece of it, and passes it forward.
    """
    player_name: str
    metrics: PlayerMetrics
    current_location: str
    inventory: list[str]
    turn_count: int
    max_turns: int
    difficulty_threshold: int
    
    # Annotated with operator.add means every time a node returns a new string, 
    # it appends it to the list instead of overwriting the history.
    narrative_history: Annotated[list[str], operator.add]
    
    # The immediate text to display to the user on the current turn
    latest_scene_text: str 
    
    # A flag to trigger a game-over or completion state
    is_active: bool    
    
    # ---> НОВЫЕ ПЕРЕМЕННЫЕ ДЛЯ СОХРАНЕНИЯ НАСТРОЕК <---
    language: str
    player_gender: str
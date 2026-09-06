from typing import TypedDict, Annotated, Optional
import operator

class PlayerMetrics(TypedDict):
    tech: int
    charm: int
    fitness: int
    intellect: int
    combat: int

class EngineState(TypedDict):
    """
    The global dictionary that LangGraph passes from node to node.
    """
    player_name: str
    metrics: PlayerMetrics
    current_location: str
    inventory: list[str]
    turn_count: int
    max_turns: int
    difficulty_threshold: int
    
    narrative_history: Annotated[list[str], operator.add]
    latest_scene_text: str 
    is_active: bool    
    
    language: str
    player_gender: str
    latest_image_url: Optional[str]  # Безопасно для любой версии Python
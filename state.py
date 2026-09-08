from typing import TypedDict, Annotated, Optional
import operator


class PlayerMetrics(TypedDict):
    tech: int
    charm: int
    fitness: int
    intellect: int
    combat: int


class EngineState(TypedDict):
    """Global state LangGraph passes from node to node."""
    player_name: str
    metrics: PlayerMetrics
    current_location: str
    inventory: list[str]
    turn_count: int
    max_turns: int
    difficulty_threshold: int
    narrative_history: Annotated[list[str], operator.add]
    turn_log: Annotated[list[dict], operator.add]   # one {stat, success, probed} per turn, for the finale
    latest_scene_text: str
    is_active: bool
    language: str
    player_gender: str
    current_scenario: str                           # the full opening scene text, kept for the finale
    skip_image: bool                                 # per-request: frontend can ask to skip image gen (local testing)
    latest_image_url: Optional[str]

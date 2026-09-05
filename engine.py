import os
from dotenv import load_dotenv
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from state import EngineState, PlayerMetrics

# Initialize the AI with Instructor
load_dotenv()
client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

class GameTurnOutput(BaseModel):
    stat_tested: str = Field(description="The primary stat attribute name being used by the player's action, e.g. 'charm', 'tech', 'combat', 'stress_tolerance', etc.")
    success: bool = Field(description="Calculated automatically: True if player stat >= current difficulty threshold, False if player stat < current difficulty threshold.")
    narrative_text: str = Field(description="The cinematic narrative continuation. If success is True, write a victory/progress. If success is False, write a struggle/setback.")
    stat_upgraded: str | None = Field(default=None, description="If success is True, name the stat to upgrade (+1), otherwise null.")
    items_added: list[str] = Field(default=[], description="Any items acquired this turn.")
    items_removed: list[str] = Field(default=[], description="Any items lost or used this turn.")

def game_master_node(state: EngineState):
    raw_metrics = state["metrics"]
    metrics = PlayerMetrics(**raw_metrics) if isinstance(raw_metrics, dict) else raw_metrics

    turn = state.get("turn_count", 1)
    max_turns = state.get("max_turns", 10)
    current_difficulty = 3 + (turn // 2) 

    metrics_dict = metrics.model_dump()
    sorted_metrics = sorted(metrics_dict.items(), key=lambda x: x[1])
    weakest_stat = sorted_metrics[0][0]
    strongest_stat = sorted_metrics[-1][0]

    print(f"[DEBUG] Turn: {turn}/{max_turns} | Difficulty: {current_difficulty}")

    recent_history = "\n\n".join(state["narrative_history"][-4:])
    current_inventory = state.get("inventory", [])

    system_prompt = f"""
    You are the Game Master of a gritty, high-stakes interactive fiction.
    It is Turn {turn} of {max_turns}. Current Difficulty Threshold: {current_difficulty}.
    
    PLAYER STATS:
    - Tech: {metrics.tech}
    - Charm: {metrics.charm}
    - Fitness: {metrics.fitness}
    - Intellect: {metrics.intellect}
    - Combat: {metrics.combat}
    - Cautiousness: {metrics.cautiousness}
    - Wealth: {metrics.wealth}
    - Alignment: {metrics.alignment}
    - Language: {metrics.language}
    - Stress Tolerance: {metrics.stress_tolerance}
    - Presence: {metrics.presence}

    RULES FOR EVALUATION:
    1. Read the Recent History. Identify the player's attempted action and determine which stat they are using (`stat_tested`).
    2. LOOK UP THE MATH: Find the value of `stat_tested` in the Player Stats above. 
    3. STRICT THRESHOLD RULE: 
       - If Stat Value >= {current_difficulty}, `success` MUST be `true`.
       - If Stat Value < {current_difficulty}, `success` MUST be `false`.
    4. Write the `narrative_text` to strictly match that outcome (`success = true` means they pull it off or gain ground; `success = false` means they fail or face a setback).
    5. Keep responses under 3 paragraphs. Crisp, sharp, cinematic English.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=GameTurnOutput,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"RECENT HISTORY:\n{recent_history}\n\nEvaluate action and render the scene:"}
        ]
    )
    
    # --- IRONCLAD PYTHON MATH OVERRIDE ---
    # Even if the AI drifts, Python checks the exact numbers here
    stat_val = metrics_dict.get(response.stat_tested, 3)
    if stat_val >= current_difficulty:
        response.success = True
        response.stat_upgraded = response.stat_tested
    else:
        response.success = False
        response.stat_upgraded = None

    # Apply stat upgrade if successful
    if response.success and response.stat_upgraded:
        current_val = getattr(metrics, response.stat_upgraded, 3)
        setattr(metrics, response.stat_upgraded, current_val + 1)

    # Structurally update inventory
    updated_inventory = list(current_inventory)
    for item in response.items_added:
        clean_item = item.strip()
        if clean_item and clean_item not in updated_inventory:
            updated_inventory.append(clean_item)
    for item in response.items_removed:
        clean_item = item.strip()
        if clean_item in updated_inventory:
            updated_inventory.remove(clean_item)

    new_history_entry = f"\nSYSTEM: {response.narrative_text}\n"

    return {
        "narrative_history": state["narrative_history"] + [new_history_entry],
        "turn_count": turn + 1,
        "metrics": metrics,
        "inventory": updated_inventory,
        "difficulty_threshold": current_difficulty
    }

def finale_node(state: EngineState):
    raw_metrics = state["metrics"]
    metrics = PlayerMetrics(**raw_metrics) if isinstance(raw_metrics, dict) else raw_metrics
    recent_history = "\n\n".join(state["narrative_history"][-3:])
    
    system_prompt = "You are the Game Master. The simulation is ending. Resolve the story definitively based on their journey."
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=GameTurnOutput,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"RECENT HISTORY:\n{recent_history}\n\nRender the finale:"}
        ]
    )
    
    return {
        "narrative_history": state["narrative_history"] + [f"\nFINALE: {response.narrative_text}\n"],
        "is_active": False  
    }

def router_node(state: EngineState):
    return {}

def route_next(state: EngineState) -> str:
    if state.get("turn_count", 1) >= state.get("max_turns", 10):
        return "finale"
    return "game_master"

workflow = StateGraph(EngineState)
workflow.add_node("router", router_node)
workflow.add_node("game_master", game_master_node)
workflow.add_node("finale", finale_node)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_next, {"game_master": "game_master", "finale": "finale"})
workflow.add_edge("game_master", END)
workflow.add_edge("finale", END)
narrative_engine = workflow.compile()
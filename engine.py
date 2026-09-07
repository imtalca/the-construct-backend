import os
from dotenv import load_dotenv
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from state import EngineState

load_dotenv()
base_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = instructor.from_openai(base_client)

class GameTurnOutput(BaseModel):
    stat_tested: str = Field(description="The primary stat attribute name being used by the player's action, e.g. 'charm', 'tech', 'combat', 'stress_tolerance', etc.")
    success: bool = Field(description="Calculated automatically: True if player stat >= current difficulty threshold, False if player stat < current difficulty threshold.")
    narrative_text: str = Field(description="The cinematic narrative continuation in the requested language, written as a single short paragraph.")
    image_prompt: str = Field(description="A highly detailed, cinematic visual description of the current scene IN ENGLISH. MUST BE SAFE FOR WORK. NO violence, NO weapons, NO blood, NO combat. Focus ONLY on the atmospheric environment, sci-fi architecture, and lighting (e.g., 'wide shot, empty glowing corridors').")
    stat_upgraded: str | None = Field(default=None, description="If success is True, name the stat to upgrade (+1), otherwise null.")
    items_added: list[str] = Field(default=[], description="Any items acquired this turn.")
    items_removed: list[str] = Field(default=[], description="Any items lost or used this turn.")

def game_master_node(state: EngineState):
    raw_metrics = state.get("metrics") or {}
    metrics = raw_metrics.model_dump() if hasattr(raw_metrics, "model_dump") else dict(raw_metrics)

    turn = state.get("turn_count", 1)
    max_turns = state.get("max_turns", 15)
    current_difficulty = 3 + (turn // 2)

    player_gender = state.get("player_gender", "Unspecified")
    player_lang = state.get("language", "en")

    lang_mapping = {'en': 'English', 'fr': 'French', 'de': 'German', 'ru': 'Russian'}
    target_language = lang_mapping.get(player_lang, 'English')

    m_tech = metrics.get('tech', 3)
    m_charm = metrics.get('charm', 3)
    m_fitness = metrics.get('fitness', 3)
    m_intellect = metrics.get('intellect', 3)
    m_combat = metrics.get('combat', 3)

    print(f"[DEBUG] Turn: {turn}/{max_turns} | Difficulty: {current_difficulty} | Lang: {target_language}")

    recent_history = "\n\n".join(state["narrative_history"][-4:])
    current_inventory = state.get("inventory", [])

    system_prompts_by_lang = {
        'en': f"""STRICT LANGUAGE: `narrative_text` IN ENGLISH ONLY. FORMAT: ONE COMPACT PARAGRAPH.
        Turn {turn}/{max_turns}. Difficulty: {current_difficulty}. Gender: {player_gender}.
        Stats: Tech:{m_tech}, Charm:{m_charm}, Fitness:{m_fitness}, Intellect:{m_intellect}, Combat:{m_combat}.
        RULES: 1) Eval stat. 2) Success if Stat >= {current_difficulty}. 3) TAKE AGENCY: Introduce new obstacle or reason to act sometimes!""",
        
        'fr': f"""EXIGENCE LINGUISTIQUE: `narrative_text` EN FRANÇAIS SEULEMENT. FORMAT: UN SEUL PARAGRAPHE COMPACT.
        Tour {turn}/{max_turns}. Difficulté: {current_difficulty}. Genre: {player_gender}.
        Stats: Tech:{m_tech}, Charm:{m_charm}, Combat:{m_combat}.
        RÈGLES: 1) Évaluez. 2) Succès si Stat >= {current_difficulty}. 3) FAITES AVANCER: Introduisez parfois un nouvel obstacle ou une raison pour une action!""",
        
        'de': f"""STRIKTE SPRACHANFORDERUNG: `narrative_text` NUR AUF DEUTSCH. FORMAT: EIN KOMPAKTER ABSATZ.
        Runde {turn}/{max_turns}. Schwelle: {current_difficulty}. Geschlecht: {player_gender}.
        Werte: Tech:{m_tech}, Combat:{m_combat}.
        REGELN: 1) Auswerten. 2) Erfolg wenn >= {current_difficulty}. 3) NEUES HINDERNIS oder Grund für das Handeln manchmal einführen!""",
        
        'ru': f"""СТРОГОЕ ТРЕБОВАНИЕ: `narrative_text` ТОЛЬКО НА РУССКОМ. ФОРМАТ: ОДИН КОМПАКТНЫЙ АБЗАЦ.
        Ход {turn}/{max_turns}. Сложность: {current_difficulty}. Пол: {player_gender}.
        Характеристики: Tech:{m_tech}, Charm:{m_charm}, Combat:{m_combat}.
        ПРАВИЛА: 1) Оцени. 2) Успех если >= {current_difficulty}. 3) БЕРИ ИНИЦИАТИВУ: периодически вводи новое препятствие или причину для действия!"""
    }

    user_prompts_by_lang = {
        'en': f"RECENT HISTORY:\n{recent_history}\n\nEvaluate action, reveal consequence, add new event (one paragraph). Also generate `image_prompt` (atmospheric only, no violence):",
        'fr': f"HISTORIQUE:\n{recent_history}\n\nÉvaluez l'action, ajoutez un événement (un paragraphe). Générez aussi `image_prompt` en anglais (pas de violence) :",
        'de': f"GESCHICHTE:\n{recent_history}\n\nAktion auswerten, neues Ereignis (ein Absatz). Auch `image_prompt` auf Englisch generieren (keine Gewalt):",
        'ru': f"ИСТОРИЯ:\n{recent_history}\n\nОцени, добавь новое событие (один абзац). Также сгенерируй `image_prompt` на английском (только атмосфера, без жестокости):"
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=GameTurnOutput,
        max_tokens=500,
        messages=[
            {"role": "system", "content": system_prompts_by_lang.get(player_lang, system_prompts_by_lang['en'])},
            {"role": "user", "content": user_prompts_by_lang.get(player_lang, user_prompts_by_lang['en'])}
        ]
    )

    image_url = None
    try:
        print(f"[DEBUG] Generating image: {response.image_prompt}")
        image_res = base_client.images.generate(
            model="gpt-image-2",
            prompt=response.image_prompt,
            size="1024x1024",
            quality="low",
            n=1,
        )
        img_item = image_res.data[0]
        # gpt-image returns base64 (b64_json); dall-e returns a hosted url
        if getattr(img_item, "url", None):
            image_url = img_item.url
        elif getattr(img_item, "b64_json", None):
            image_url = f"data:image/png;base64,{img_item.b64_json}"
        print("[DEBUG] Image generated.")
    except Exception as e:
        print(f"[ERROR] Failed to generate image: {e}")
        image_url = None

    stat_val = metrics.get(response.stat_tested, 3)
    if stat_val >= current_difficulty:
        response.success = True
        response.stat_upgraded = response.stat_tested
    else:
        response.success = False
        response.stat_upgraded = None

    if response.success and response.stat_upgraded:
        key = response.stat_upgraded
        metrics[key] = metrics.get(key, 3) + 1

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
        # operator.add reducer on this channel -> return only the new entry
        "narrative_history": [new_history_entry],
        "turn_count": turn + 1,
        "metrics": metrics,
        "inventory": updated_inventory,
        "difficulty_threshold": current_difficulty,
        "max_turns": max_turns,
        "language": player_lang,
        "player_gender": player_gender,
        "latest_image_url": image_url,
    }

def finale_node(state: EngineState):
    recent_history = "\n\n".join(state["narrative_history"][-3:])
    player_lang = state.get("language", "en")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=GameTurnOutput,
        max_tokens=300,
        messages=[
            {"role": "system", "content": f"WRITE IN {player_lang.upper()} ONLY. Resolve the story."},
            {"role": "user", "content": f"HISTORY:\n{recent_history}\n\nRender the finale:"}
        ]
    )
    
    return {
        "narrative_history": [f"\nFINALE: {response.narrative_text}\n"],
        "is_active": False,
        "latest_image_url": None
    }

def router_node(state: EngineState):
    return {}

def route_next(state: EngineState) -> str:
    max_t = state.get("max_turns", 15)
    if state.get("turn_count", 1) >= max_t:
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
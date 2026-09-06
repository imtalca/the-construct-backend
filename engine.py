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
    narrative_text: str = Field(description="The cinematic narrative continuation in the requested language, written as a single short paragraph.")
    stat_upgraded: str | None = Field(default=None, description="If success is True, name the stat to upgrade (+1), otherwise null.")
    items_added: list[str] = Field(default=[], description="Any items acquired this turn.")
    items_removed: list[str] = Field(default=[], description="Any items lost or used this turn.")

def game_master_node(state: EngineState):
    raw_metrics = state["metrics"]
    metrics = PlayerMetrics(**raw_metrics) if isinstance(raw_metrics, dict) else raw_metrics

    turn = state.get("turn_count", 1)
    max_turns = state.get("max_turns", 15)  
    current_difficulty = 3 + (turn // 2) 

    # Extract user profile data sent from frontend
    player_gender = state.get("player_gender", "Unspecified")
    player_lang = state.get("language", "en")
    
    lang_mapping = {
        'en': 'English',
        'fr': 'French',
        'de': 'German',
        'ru': 'Russian'
    }
    target_language = lang_mapping.get(player_lang, 'English')

    metrics_dict = metrics.model_dump()

    print(f"[DEBUG] Turn: {turn}/{max_turns} | Difficulty: {current_difficulty} | Lang: {target_language}")

    recent_history = "\n\n".join(state["narrative_history"][-4:])
    current_inventory = state.get("inventory", [])

    # Локализованные системные промпты (решают проблему срыва на английский)
    system_prompts_by_lang = {
        'en': f"""
        STRICT LANGUAGE REQUIREMENT: YOU MUST WRITE ALL `narrative_text` EXCLUSIVELY, 100%, AND ENTIRELY IN ENGLISH. NO OTHER LANGUAGE.
        FORMAT REQUIREMENT: Write the `narrative_text` as a SINGLE, SHORT, COMPACT PARAGRAPH (1-3 sentences max) with NO line breaks. Max tokens limit applies.
        
        You are the Game Master of a gritty, high-stakes sci-fi interactive fiction.
        Turn {turn} of {max_turns}. Difficulty Threshold: {current_difficulty}. Player Gender: {player_gender}.
        
        PLAYER STATS: Tech: {metrics.tech}, Charm: {metrics.charm}, Fitness: {metrics.fitness}, Intellect: {metrics.intellect}, Combat: {metrics.combat}, Cautiousness: {metrics.cautiousness}, Wealth: {metrics.wealth}, Alignment: {metrics.alignment}, Language: {metrics.language}, Stress Tolerance: {metrics.stress_tolerance}, Presence: {metrics.presence}.
        
        RULES:
        1. Identify player action and `stat_tested`.
        2. If Stat Value >= {current_difficulty}, `success` = true, else false.
        3. Write `narrative_text` in English as a single short paragraph.
        """,
        'fr': f"""
        EXIGENCE LINGUISTIQUE STRICTE : VOUS DEVEZ ÉCRIRE TOUT LE `narrative_text` EXCLUSIVEMENT ET EN ENTIER EN FRANÇAIS. AUCUNE AUTRE LANGUE.
        FORMAT : Écrivez `narrative_text` EN UN SEUL PARAGRAPHE COURT ET COMPACT (1-3 phrases max) sans sauts de ligne.
        
        Vous êtes le Maître du Jeu d'une fiction interactive de science-fiction.
        Tour {turn} sur {max_turns}. Seuil de difficulté : {current_difficulty}. Genre du joueur : {player_gender}.
        
        STATS DU JOUEUR : Tech: {metrics.tech}, Charm: {metrics.charm}, Fitness: {metrics.fitness}, Intellect: {metrics.intellect}, Combat: {metrics.combat}, Cautiousness: {metrics.cautiousness}, Wealth: {metrics.wealth}, Alignment: {metrics.alignment}, Language: {metrics.language}, Stress Tolerance: {metrics.stress_tolerance}, Presence: {metrics.presence}.
        
        RÈGLES :
        1. Identifiez l'action et le `stat_tested`.
        2. Si Stat >= {current_difficulty}, `success` = true, sinon false.
        3. Rédigez `narrative_text` en français en un seul paragraphe court.
        """,
        'de': f"""
        STRIKTE SPRACHANFORDERUNG: SIE MÜSSEN GESAMTEN `narrative_text` AUSSCHLIESSLICH UND VOLLSTÄNDIG AUF DEUTSCH SCHREIBEN. KEINE ANDERE SPRACHE.
        FORMAT: Schreiben Sie `narrative_text` als EINZIGEN, KURZEN, KOMPAKTEN ABSATZ (max. 1-3 Sätze) ohne Zeilenumbrüche.
        
        Sie sind der Spielleiter einer gritty Sci-Fi-Interactive-Fiction.
        Runde {turn} von {max_turns}. Schwierigkeitsschwelle: {current_difficulty}. Spielergeschlecht: {player_gender}.
        
        SPIELERWERTE: Tech: {metrics.tech}, Charm: {metrics.charm}, Fitness: {metrics.fitness}, Intellect: {metrics.intellect}, Combat: {metrics.combat}, Cautiousness: {metrics.cautiousness}, Wealth: {metrics.wealth}, Alignment: {metrics.alignment}, Language: {metrics.language}, Stress Tolerance: {metrics.stress_tolerance}, Presence: {metrics.presence}.
        
        REGELN:
        1. Identifizieren Sie die Aktion und `stat_tested`.
        2. Wenn Wert >= {current_difficulty}, `success` = true, sonst false.
        3. Schreiben Sie `narrative_text` auf Deutsch in einem kurzen Absatz.
        """,
        'ru': f"""
        СТРОГОЕ ТРЕБОВАНИЕ К ЯЗЫКУ: ВЫ ОБЯЗАНЫ НАПИСАТЬ ВЕСЬ ТЕКСТ В ПОЛЕ `narrative_text` ИСКЛЮЧИТЕЛЬНО, НА 100% И ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. НИКАКОГО АНГЛИЙСКОГО.
        ФОРМАТ: Пишите `narrative_text` ОДНИМ КОРОТКИМ, КОМПАКТНЫМ АБЗАЦЕМ (1-3 предложения максимум) без переносов строк.
        
        Ты — Мастер Игры (Game Master) в жесткой научно-фантастической текстовой ролевой игре.
        Ход {turn} из {max_turns}. Порог сложности: {current_difficulty}. Пол игрока: {player_gender}.
        
        ХАРАКТЕРИСТИКИ ИГРОКА: Tech: {metrics.tech}, Charm: {metrics.charm}, Fitness: {metrics.fitness}, Intellect: {metrics.intellect}, Combat: {metrics.combat}, Cautiousness: {metrics.cautiousness}, Wealth: {metrics.wealth}, Alignment: {metrics.alignment}, Language: {metrics.language}, Stress Tolerance: {metrics.stress_tolerance}, Presence: {metrics.presence}.
        
        ПРАВИЛА:
        1. Определи действие игрока и проверяемую характеристику (`stat_tested`).
        2. Если значение характеристики >= {current_difficulty}, то `success` = true, иначе false.
        3. Напиши `narrative_text` строго на русском языке в виде одного короткого абзаца.
        """
    }

    system_prompt = system_prompts_by_lang.get(player_lang, system_prompts_by_lang['en'])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=GameTurnOutput,
        max_tokens=300,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"RECENT HISTORY:\n{recent_history}\n\nEvaluate action and render the scene strictly in {target_language} as one short paragraph:"}
        ]
    )
    
    # --- IRONCLAD PYTHON MATH OVERRIDE ---
    stat_val = metrics_dict.get(response.stat_tested, 3)
    if stat_val >= current_difficulty:
        response.success = True
        response.stat_upgraded = response.stat_tested
    else:
        response.success = False
        response.stat_upgraded = None

    if response.success and response.stat_upgraded:
        current_val = getattr(metrics, response.stat_upgraded, 3)
        setattr(metrics, response.stat_upgraded, current_val + 1)

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
        "difficulty_threshold": current_difficulty,
        "max_turns": max_turns,
        "language": player_lang,
        "player_gender": player_gender
    }

def finale_node(state: EngineState):
    raw_metrics = state["metrics"]
    metrics = PlayerMetrics(**raw_metrics) if isinstance(raw_metrics, dict) else raw_metrics
    recent_history = "\n\n".join(state["narrative_history"][-3:])
    player_lang = state.get("language", "en")
    
    lang_mapping = {'en': 'English', 'fr': 'French', 'de': 'German', 'ru': 'Russian'}
    target_language = lang_mapping.get(player_lang, 'English')
    
    system_prompts_finale = {
        'en': "STRICT LANGUAGE REQUIREMENT: WRITE ENTIRELY IN ENGLISH. Single short paragraph. Resolve the story definitively.",
        'fr': "EXIGENCE LINGUISTIQUE : ÉCRIRE EN FRANÇAIS. Un seul paragraphe court. Résolvez l'histoire.",
        'de': "SPRACHANFORDERUNG: VOLLSTÄNDIG AUF DEUTSCH SCHREIBEN. Ein kurzer Absatz. Beende die Geschichte.",
        'ru': "СТРОГОЕ ТРЕБОВАНИЕ К ЯЗЫКУ: ПИШИТЕ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. Один короткий абзац. Завершите историю."
    }
    system_prompt = system_prompts_finale.get(player_lang, system_prompts_finale['en'])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=GameTurnOutput,
        max_tokens=300,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"RECENT HISTORY:\n{recent_history}\n\nRender the finale in {target_language}:"}
        ]
    )
    
    return {
        "narrative_history": state["narrative_history"] + [f"\nFINALE: {response.narrative_text}\n"],
        "is_active": False  
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
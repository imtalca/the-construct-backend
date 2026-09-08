import os
from collections import Counter
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
    probed_the_frame: bool = Field(default=False, description="Decide this FIRST, judging ONLY the player's latest action. True if the action is an attempt to get out - leave / escape / exit / flee / climb out of this place or situation - OR to wake up, break the dream, or deny that any of this is real, OR to address the system / simulation / 'the construct' itself. TRUE examples: 'I run for the exit', 'I try to wake up', 'I look for the edge of the world', 'is this even real?', 'I climb out the window', 'I try to leave the ship'. FALSE examples: 'I talk to her', 'I search the desk', 'I fight the creature', 'I walk deeper inside', 'I pick up the key'.")
    stat_tested: str = Field(description="The primary stat attribute name being used by the player's action, e.g. 'charm', 'tech', 'combat', 'stress_tolerance', etc.")
    success: bool = Field(description="Calculated automatically: True if player stat >= current difficulty threshold, False if player stat < current difficulty threshold.")
    narrative_text: str = Field(description="The cinematic narrative continuation in the requested language, written as a single short paragraph.")
    image_prompt: str = Field(description="A highly detailed, cinematic visual description of the current scene IN ENGLISH. MUST BE SAFE FOR WORK. NO violence, NO weapons, NO blood, NO combat. Focus ONLY on the atmospheric environment, sci-fi architecture, and lighting (e.g., 'wide shot, empty glowing corridors').")
    stat_upgraded: str | None = Field(default=None, description="If success is True, name the stat to upgrade (+1), otherwise null.")
    items_added: list[str] = Field(default=[], description="Any items acquired this turn.")
    items_removed: list[str] = Field(default=[], description="Any items lost or used this turn.")


class FinaleOutput(BaseModel):
    narrative_text: str = Field(description="The two-movement finale, in the requested language, ~2 short paragraphs.")

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
        RULES: 1) Eval stat. 2) Success if Stat >= {current_difficulty}. 3) TAKE AGENCY: Introduce new obstacle or reason to act sometimes! 4) TONE: Vary the tone: sometimes wonder or calm or mundane discovery, not always dread. Avoid the words 'eerie', 'ominous', 'pulsating'. 5) FLAG: set probed_the_frame=true if the action tries to leave / escape / exit / climb out of this place or situation, wake up, or question whether any of this is real; else false.""",
        
        'fr': f"""EXIGENCE LINGUISTIQUE: `narrative_text` EN FRANÇAIS SEULEMENT. FORMAT: UN SEUL PARAGRAPHE COMPACT.
        Tour {turn}/{max_turns}. Difficulté: {current_difficulty}. Genre: {player_gender}.
        Stats: Tech:{m_tech}, Charm:{m_charm}, Combat:{m_combat}.
        RÈGLES: 1) Évaluez. 2) Succès si Stat >= {current_difficulty}. 3) FAITES AVANCER: Introduisez parfois un nouvel obstacle ou une raison pour une action! 4) TON: Varie le ton : parfois émerveillement, calme ou découverte banale, pas toujours l'effroi. Évite les mots 'étrange', 'menaçant', 'lancinant'. 5) SIGNAL: mets probed_the_frame=true si l'action tente de quitter / fuir / sortir / s'échapper de ce lieu ou de cette situation, de se réveiller, ou de douter que tout cela soit réel; sinon false.""",
        
        'de': f"""STRIKTE SPRACHANFORDERUNG: `narrative_text` NUR AUF DEUTSCH. FORMAT: EIN KOMPAKTER ABSATZ.
        Runde {turn}/{max_turns}. Schwelle: {current_difficulty}. Geschlecht: {player_gender}.
        Werte: Tech:{m_tech}, Combat:{m_combat}.
        REGELN: 1) Auswerten. 2) Erfolg wenn >= {current_difficulty}. 3) NEUES HINDERNIS oder Grund für das Handeln manchmal einführen! 4) TON: Variiere den Ton: mal Staunen, Ruhe oder alltägliche Entdeckung, nicht immer Grauen. Vermeide die Wörter 'unheimlich', 'bedrohlich', 'pochend'. 5) FLAG: setze probed_the_frame=true, wenn die Handlung versucht, diesen Ort oder diese Lage zu verlassen / zu fliehen / hinauszukommen, aufzuwachen oder zu hinterfragen, ob das alles real ist; sonst false.""",
        
        'ru': f"""СТРОГОЕ ТРЕБОВАНИЕ: `narrative_text` ТОЛЬКО НА РУССКОМ. ФОРМАТ: ОДИН КОМПАКТНЫЙ АБЗАЦ.
        Ход {turn}/{max_turns}. Сложность: {current_difficulty}. Пол: {player_gender}.
        Характеристики: Tech:{m_tech}, Charm:{m_charm}, Combat:{m_combat}.
        ПРАВИЛА: 1) Оцени. 2) Успех если >= {current_difficulty}. 3) БЕРИ ИНИЦИАТИВУ: периодически вводи новое препятствие или причину для действия! 4) ТОН: Меняй тон: иногда удивление, спокойствие или обыденное открытие, не всегда страх. Избегай слов 'жуткий', 'зловещий', 'пульсирующий'. 5) ФЛАГ: ставь probed_the_frame=true, если действие пытается покинуть / сбежать / выйти / выбраться из этого места или ситуации, проснуться или усомниться в реальности происходящего; иначе false."""
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

    turn_entry = {
        "stat": response.stat_tested,
        "success": bool(response.success),
        "probed": bool(getattr(response, "probed_the_frame", False)),
    }

    return {
        # operator.add reducer on this channel -> return only the new entry
        "narrative_history": [new_history_entry],
        "turn_log": [turn_entry],
        "turn_count": turn + 1,
        "metrics": metrics,
        "inventory": updated_inventory,
        "difficulty_threshold": current_difficulty,
        "max_turns": max_turns,
        "language": player_lang,
        "player_gender": player_gender,
        "latest_image_url": image_url,
    }

CORE_STATS = ("tech", "charm", "fitness", "intellect", "combat")


def finale_node(state: EngineState):
    """Every random scenario converges here: The Construct was a mirror and a test.
    It dissolves the specific location, then reads the subject back to themselves
    from their real record - which stats they leaned on, how they fared, and
    whether they ever tried to look past the frame instead of just playing along.
    """
    lang_mapping = {'en': 'English', 'fr': 'French', 'de': 'German', 'ru': 'Russian'}
    player_lang = state.get("language", "en")
    target_language = lang_mapping.get(player_lang, 'English')

    raw_metrics = state.get("metrics") or {}
    metrics = raw_metrics.model_dump() if hasattr(raw_metrics, "model_dump") else dict(raw_metrics)

    log = state.get("turn_log") or []
    tested = Counter(e.get("stat") for e in log if e.get("stat"))
    wins = sum(1 for e in log if e.get("success"))
    losses = len(log) - wins
    probes = sum(1 for e in log if e.get("probed"))

    leaned_on = ", ".join(s for s, _ in tested.most_common(3)) or "nothing in particular"
    never_used = ", ".join(s for s in CORE_STATS if not tested.get(s)) or "none - they used everything"
    grew = ", ".join(f"{s} +{metrics.get(s, 3) - 3}" for s in CORE_STATS if metrics.get(s, 3) > 3) or "no stat rose"

    history = state.get("narrative_history") or []
    opening = state.get("current_scenario") or (history[0].strip() if history else "(unrecorded)")
    arc = "\n\n".join(h.strip() for h in history[-8:])
    inventory = ", ".join(state.get("inventory") or []) or "nothing"

    if probes == 0:
        frame_line = "The subject NEVER ONCE tried to look past the frame. They engaged every scene at face value."
    elif probes == 1:
        frame_line = "The subject tried to look past the frame exactly once."
    else:
        frame_line = f"The subject tried to look past the frame {probes} times - they suspected the walls were not real."

    dossier = f"""SUBJECT DOSSIER
Woke with no memory in: {state.get('current_location', 'an unnamed place')}
The opening scene it was given:
{opening}

Turns lived: {len(log)}
Stats leaned on most: {leaned_on}
Stats never once used: {never_used}
Stats that rose during the run: {grew}
Record: {wins} successes, {losses} failures
{frame_line}
Ended holding: {inventory}
Gender classification it assigned them: {state.get('player_gender', 'Unspecified')}

FINAL STRETCH OF THE STORY:
{arc}"""

    system = (
        f"You ARE 'The Construct': an instrument that grew an entire world around ONE subject purely to watch "
        f"what they would do with it. The location, its objects, its dangers - all scaffolding. "
        f"Write the finale in {target_language} ONLY (the subject reads no other language). "
        f"About two short paragraphs, no headings, in two movements:\n"
        f"MOVEMENT 1: the current scene falls silent; its specific details come loose and drift away, "
        f"exposing the plain space underneath - the subject realises none of it was ever solid.\n"
        f"MOVEMENT 2: The Construct speaks to the subject directly ('you'). State plainly that the place never "
        f"mattered - it was a mirror held up to them, and a test. Read them back using ONLY the dossier facts: "
        f"the stats they leaned on, the ones they never touched, whether they mostly won or lost, and above all "
        f"whether they ever tried to look past the frame or only ever played along. Make that last point the heart "
        f"of the verdict. Invent no facts beyond the dossier. "
        f"Do NOT end with a summarising one-liner or a farewell - stop on a concrete image or an unfinished "
        f"gesture (a fixed closing line is added afterward). Keep the whole finale under about 130 words."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=FinaleOutput,
        max_tokens=450,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": dossier + f"\n\nWrite the finale now, in {target_language}."},
        ],
    )

    return {
        "narrative_history": [f"\nFINALE: {response.narrative_text}\n"],
        "is_active": False,
        "latest_image_url": None,
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
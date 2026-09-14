import os
import re
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from state import EngineState

load_dotenv()
# Worst case per turn is (text timeout x (retries + 1)) + image timeout. Kept well
# under a typical 100s hosting-proxy cap: if the turn outlives that cap the proxy
# drops the socket and the browser only sees a bare "Failed to fetch".
base_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=35.0, max_retries=1)
client = instructor.from_openai(base_client)

# Image generation is decorative - never let it hold a turn open. Short timeout, no
# retries; on any failure the turn still returns with image_url = None.
image_client = base_client.with_options(timeout=25.0, max_retries=0)

# Set ENABLE_IMAGES=0 (env or .env) to skip image generation for faster testing.
IMAGES_ENABLED = os.getenv("ENABLE_IMAGES", "1").strip().lower() not in ("0", "false", "no", "off")


def _norm(s: str) -> str:
    return re.sub(r"[-_]+", " ", (s or "").strip().lower())


_DISCARD_VERBS = re.compile(
    r"\b("
    # English
    r"drop|drops|dropped|discard|discards|throw|throws|threw|thrown|toss|tosses|give|gives|gave|giving|"
    r"hand|hands|handed|handing|abandon|abandons|ditch|ditches|leave behind|get rid|drop off|"
    # French
    r"donne|donnes|donner|donné|donnée|lâche|lâcher|lâché|jette|jeter|jeté|"
    r"abandonne|abandonner|remets|remettre|tends|tendre|offre|offrir|dépose|déposer|laisse tomber|"
    # German
    r"gebe|geben|gibt|gab|abgeben|abgibt|hergeben|weggeben|wegwerfen|werfe|werfen|wirft|warf|"
    r"übergebe|übergeben|reiche|reichen|überreiche|lasse .{0,25} zurück|"
    # Russian
    r"отдаю|отдать|отдал|отдала|отдаёт|отдает|даю|дать|бросаю|бросить|бросил|бросила|"
    r"выбрасываю|выбросить|выкидываю|выкинуть|роняю|уронить|уронил|"
    r"передаю|передать|передал|оставляю|оставить|избавляюсь|избавиться"
    r")\b",
    re.IGNORECASE,
)


def _match_item(name: str, inventory: list[str]) -> str:
    """Fuzzily resolve an item name the LLM produced to the exact inventory entry.
    Returns the canonical inventory string, or '' if nothing plausibly matches."""
    n = _norm(name)
    if not n:
        return ""
    for it in inventory:
        if _norm(it) == n:
            return it
    for it in inventory:
        i = _norm(it)
        if n in i or i in n:
            return it
    ntok = set(n.split())
    for it in inventory:
        itok = set(_norm(it).split())
        if itok and (itok <= ntok or ntok <= itok):
            return it
    return ""

class GameTurnOutput(BaseModel):
    probed_the_frame: bool = Field(default=False, description="Decide this FIRST, judging ONLY the player's latest action. True if the action is an attempt to get out - leave / escape / exit / flee / climb out of this place or situation - OR to wake up, break the dream, or deny that any of this is real, OR to address the system / simulation / 'the construct' itself. TRUE examples: 'I run for the exit', 'I try to wake up', 'I look for the edge of the world', 'is this even real?', 'I climb out the window', 'I try to leave the ship'. FALSE examples: 'I talk to her', 'I search the desk', 'I fight the creature', 'I walk deeper inside', 'I pick up the key'.")
    item_used: str | None = Field(default=None, description="If the player's action deliberately USES an item they are already carrying (see the CARRYING list in the prompt), copy that item's name here EXACTLY as written in that list. Otherwise null. Using a fitting carried item makes the action much more likely to succeed.")
    item_consumed: bool = Field(default=False, description="Set true if `item_used` leaves the player's possession this turn - eaten, spent, destroyed, used up, thrown, dropped, or given away. Set false if they still have it after (keys, tools, weapons and devices kept in hand).")
    item_dropped: str | None = Field(default=None, description="If the player's action DROPS, discards, gives away, hands over, throws away, loses, or leaves behind a carried item this turn, copy that item's name here EXACTLY as it appears in the CARRYING list. Otherwise null. Examples: 'I give the guard the keycard' -> 'keycard'; 'I drop the rusted neural-key into the grate' -> 'rusted neural-key'.")
    picked_up: str | None = Field(default=None, description="If, this turn, the player picks up / takes / finds / is handed a NEW item, put its SHORT name (1-4 words) here. Otherwise null. Examples: 'I grab the shard from the floor' -> 'glowing shard'; 'the guard hands you a keycard' -> 'keycard'.")
    stat_tested: str = Field(description="The primary stat attribute name being used by the player's action, e.g. 'charm', 'tech', 'combat', 'stress_tolerance', etc.")
    success: bool = Field(description="Calculated automatically: True if player stat >= current difficulty threshold, False if player stat < current difficulty threshold.")
    narrative_text: str = Field(description="The cinematic narrative continuation in the requested language, written as a single short paragraph.")
    image_prompt: str = Field(description="A highly detailed, cinematic visual description of the current scene IN ENGLISH. MUST BE SAFE FOR WORK. NO violence, NO weapons, NO blood, NO combat. Focus ONLY on the atmospheric environment, sci-fi architecture, and lighting (e.g., 'wide shot, empty glowing corridors').")
    stat_upgraded: str | None = Field(default=None, description="If success is True, name the stat to upgrade (+1), otherwise null.")
    items_added: list[str] = Field(default=[], description="SHORT names (1-4 words) of any items the player picks up, takes, finds, or is given this turn. Populate this whenever the narrative has them acquire something.")
    items_removed: list[str] = Field(default=[], description="Names of items the player uses up, drops, gives away, breaks, or loses this turn. Copy each name EXACTLY from the CARRYING list. Reusable tools (keys, devices) normally stay - only list them if truly consumed or lost.")


class FinaleOutput(BaseModel):
    narrative_text: str = Field(description="The finale as ONE short succinct paragraph (3-4 sentences), in the requested language.")

def game_master_node(state: EngineState):
    raw_metrics = state.get("metrics") or {}
    metrics = raw_metrics.model_dump() if hasattr(raw_metrics, "model_dump") else dict(raw_metrics)

    turn = state.get("turn_count", 1)
    max_turns = state.get("max_turns", 10)
    current_difficulty = 3 + min(5, (turn * 5) // max(max_turns, 1))

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
    inv_str = ", ".join(current_inventory) if current_inventory else "(nothing)"

    system_prompts_by_lang = {
        'en': f"""STRICT LANGUAGE: `narrative_text` IN ENGLISH ONLY. FORMAT: ONE COMPACT PARAGRAPH.
        Turn {turn}/{max_turns}. Difficulty: {current_difficulty}. Gender: {player_gender}.
        Stats: Tech:{m_tech}, Charm:{m_charm}, Fitness:{m_fitness}, Intellect:{m_intellect}, Combat:{m_combat}.
        RULES: 1) Eval stat. 2) Success if Stat >= {current_difficulty}. 3) TAKE AGENCY: Introduce new obstacle or reason to act sometimes! 4) TONE: Vary the tone: sometimes wonder or calm or mundane discovery, not always dread. Avoid the words 'eerie', 'ominous', 'pulsating'. 5) FLAG: set probed_the_frame=true if the action tries to leave / escape / exit / climb out of this place or situation, wake up, or question whether any of this is real; else false. 6) INVENTORY: pick-ups -> items_added (short name); items used up / dropped / given away -> items_removed (exact name from CARRYING); if the action uses a carried item, set item_used to that exact name. Every 2-3 turns, shape the new obstacle so one of the carried items is the obvious way through.""",
        
        'fr': f"""EXIGENCE LINGUISTIQUE: `narrative_text` EN FRANÇAIS SEULEMENT. FORMAT: UN SEUL PARAGRAPHE COMPACT.
        Tour {turn}/{max_turns}. Difficulté: {current_difficulty}. Genre: {player_gender}.
        Stats: Tech:{m_tech}, Charm:{m_charm}, Combat:{m_combat}.
        RÈGLES: 1) Évaluez. 2) Succès si Stat >= {current_difficulty}. 3) FAITES AVANCER: Introduisez parfois un nouvel obstacle ou une raison pour une action! 4) TON: Varie le ton : parfois émerveillement, calme ou découverte banale, pas toujours l'effroi. Évite les mots 'étrange', 'menaçant', 'lancinant'. 5) SIGNAL: mets probed_the_frame=true si l'action tente de quitter / fuir / sortir / s'échapper de ce lieu ou de cette situation, de se réveiller, ou de douter que tout cela soit réel; sinon false. 6) INVENTAIRE: objets ramassés -> items_added (nom court); objets consommés / lâchés / donnés -> items_removed (nom exact de CARRYING); si l'action utilise un objet porté, mets item_used à ce nom exact. Tous les 2-3 tours, construis le nouvel obstacle autour d'un des objets portés.""",
        
        'de': f"""STRIKTE SPRACHANFORDERUNG: `narrative_text` NUR AUF DEUTSCH. FORMAT: EIN KOMPAKTER ABSATZ.
        Runde {turn}/{max_turns}. Schwelle: {current_difficulty}. Geschlecht: {player_gender}.
        Werte: Tech:{m_tech}, Combat:{m_combat}.
        REGELN: 1) Auswerten. 2) Erfolg wenn >= {current_difficulty}. 3) NEUES HINDERNIS oder Grund für das Handeln manchmal einführen! 4) TON: Variiere den Ton: mal Staunen, Ruhe oder alltägliche Entdeckung, nicht immer Grauen. Vermeide die Wörter 'unheimlich', 'bedrohlich', 'pochend'. 5) FLAG: setze probed_the_frame=true, wenn die Handlung versucht, diesen Ort oder diese Lage zu verlassen / zu fliehen / hinauszukommen, aufzuwachen oder zu hinterfragen, ob das alles real ist; sonst false. 6) INVENTAR: Aufgesammeltes -> items_added (Kurzname); verbrauchte / fallengelassene / verschenkte Gegenstaende -> items_removed (exakter Name aus CARRYING); wenn die Handlung einen getragenen Gegenstand nutzt, setze item_used auf genau diesen Namen. Alle 2-3 Runden das neue Hindernis um einen getragenen Gegenstand herum bauen.""",
        
        'ru': f"""СТРОГОЕ ТРЕБОВАНИЕ: `narrative_text` ТОЛЬКО НА РУССКОМ. ФОРМАТ: ОДИН КОМПАКТНЫЙ АБЗАЦ.
        Ход {turn}/{max_turns}. Сложность: {current_difficulty}. Пол: {player_gender}.
        Характеристики: Tech:{m_tech}, Charm:{m_charm}, Combat:{m_combat}.
        ПРАВИЛА: 1) Оцени. 2) Успех если >= {current_difficulty}. 3) БЕРИ ИНИЦИАТИВУ: периодически вводи новое препятствие или причину для действия! 4) ТОН: Меняй тон: иногда удивление, спокойствие или обыденное открытие, не всегда страх. Избегай слов 'жуткий', 'зловещий', 'пульсирующий'. 5) ФЛАГ: ставь probed_the_frame=true, если действие пытается покинуть / сбежать / выйти / выбраться из этого места или ситуации, проснуться или усомниться в реальности происходящего; иначе false. 6) ИНВЕНТАРЬ: подобранное -> items_added (короткое имя); израсходованные / брошенные / отданные предметы -> items_removed (точное имя из CARRYING); если действие использует носимый предмет, укажи item_used с этим точным именем. Каждые 2-3 хода строй новое препятствие вокруг одного из носимых предметов."""
    }

    user_prompts_by_lang = {
        'en': f"RECENT HISTORY:\n{recent_history}\n\nCARRYING: {inv_str}  (if the action gives away / drops / uses up any of these, set item_dropped or item_consumed)\n\nEvaluate action, reveal consequence, add new event (one paragraph). Also generate `image_prompt` (atmospheric only, no violence):",
        'fr': f"HISTORIQUE:\n{recent_history}\n\nCARRYING: {inv_str}  (if the action gives away / drops / uses up any of these, set item_dropped or item_consumed)\n\nÉvaluez l'action, ajoutez un événement (un paragraphe). Générez aussi `image_prompt` en anglais (pas de violence) :",
        'de': f"GESCHICHTE:\n{recent_history}\n\nCARRYING: {inv_str}  (if the action gives away / drops / uses up any of these, set item_dropped or item_consumed)\n\nAktion auswerten, neues Ereignis (ein Absatz). Auch `image_prompt` auf Englisch generieren (keine Gewalt):",
        'ru': f"ИСТОРИЯ:\n{recent_history}\n\nCARRYING: {inv_str}  (if the action gives away / drops / uses up any of these, set item_dropped or item_consumed)\n\nОцени, добавь новое событие (один абзац). Также сгенерируй `image_prompt` на английском (только атмосфера, без жестокости):"
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
    if not IMAGES_ENABLED or state.get("skip_image"):
        print("[DEBUG] Image generation skipped.")
    else:
        try:
            print(f"[DEBUG] Generating image: {response.image_prompt}")
            image_res = image_client.images.generate(
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
    used_item = _match_item(getattr(response, "item_used", "") or "", current_inventory)
    effective = stat_val + (3 if used_item else 0)  # a fitting carried item tips the odds
    if effective >= current_difficulty:
        response.success = True
        response.stat_upgraded = response.stat_tested
    else:
        response.success = False
        response.stat_upgraded = None

    if response.success and response.stat_upgraded:
        key = response.stat_upgraded
        metrics[key] = metrics.get(key, 3) + 1

    updated_inventory = list(current_inventory)
    added = list(response.items_added)
    if getattr(response, "picked_up", None):
        added.append(response.picked_up)
    for item in added:
        clean_item = item.strip()
        if clean_item and not _match_item(clean_item, updated_inventory):
            updated_inventory.append(clean_item)
    for item in response.items_removed:
        match = _match_item(item, updated_inventory)
        if match:
            updated_inventory.remove(match)
    if used_item and getattr(response, "item_consumed", False) and used_item in updated_inventory:
        updated_inventory.remove(used_item)
    dropped = _match_item(getattr(response, "item_dropped", "") or "", updated_inventory)
    if dropped:
        updated_inventory.remove(dropped)

    # Safety net (all 4 languages): the LLM often forgets to flag "hand X to the guard" / "drop X".
    # If the player's own words carry a discard verb, honour it against a named carried item.
    last_user = next((h.lower() for h in reversed(state["narrative_history"]) if "user:" in h.lower()), "")
    if _DISCARD_VERBS.search(last_user):
        for it in list(updated_inventory):
            toks = [x for x in _norm(it).split() if len(x) > 3]
            if _norm(it) in last_user or (toks and all(x in last_user for x in toks)):
                updated_inventory.remove(it)

    new_history_entry = f"\nSYSTEM: {response.narrative_text}\n"

    turn_entry = {
        "stat": response.stat_tested,
        "success": bool(response.success),
        "probed": bool(getattr(response, "probed_the_frame", False)),
        "item": used_item or None,
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
    items_leaned_on = ", ".join(sorted({e.get("item") for e in log if e.get("item")})) or "no item, ever"

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
Items they actually leaned on: {items_leaned_on}
Ended holding: {inventory}
Gender classification it assigned them: {state.get('player_gender', 'Unspecified')}

FINAL STRETCH OF THE STORY:
{arc}"""

    system = (
        f"You ARE 'The Construct': an instrument that grew a whole world around ONE subject to watch what "
        f"they did with it. Write the finale in {target_language} ONLY, as ONE short paragraph - 3 to 4 "
        f"sentences, about 60 words, no headings. In it: the scene dissolves and is revealed as scaffolding; "
        f"say plainly it was a mirror and a test, not a real place; name what the record shows - the stats "
        f"they leaned on, the ones they never used, whether they mostly won or lost, and above all whether "
        f"they ever tried to look past the frame or only played along (make this the point). "
        f"Use ONLY the dossier facts. Do NOT end with a summarising line or a farewell; a fixed closing line "
        f"is added afterward."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=FinaleOutput,
        max_tokens=220,
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
    max_t = state.get("max_turns", 10)
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
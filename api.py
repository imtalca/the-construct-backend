import os
import logging
import secrets
from collections import deque
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Any
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from engine import narrative_engine, client

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("construct")

ACCESS_KEY = os.getenv("ACCESS_KEY", "")
if not ACCESS_KEY:
    print("[WARN] ACCESS_KEY is not set - all protected requests will be rejected with 401.")

# Admin feed key. Falls back to ACCESS_KEY if ADMIN_KEY is not set separately.
ADMIN_KEY = os.getenv("ADMIN_KEY", "") or ACCESS_KEY


def require_key(x_access_key: str = Header(default="")):
    supplied = (x_access_key or "").encode("utf-8")
    expected = ACCESS_KEY.encode("utf-8")
    if not ACCESS_KEY or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing access key")


def _check_admin(key: str):
    supplied = (key or "").encode("utf-8")
    expected = ADMIN_KEY.encode("utf-8")
    if not ADMIN_KEY or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing admin key")


# Recent player activity, newest last. In-memory only: cleared on every restart /
# redeploy (and Render spins the free instance down after ~15 min idle).
_ACTION_LOG: deque[dict] = deque(maxlen=500)


def _client_ip(request: Request) -> str:
    # Render / any reverse proxy puts the real client first in X-Forwarded-For;
    # request.client.host would just be the proxy's internal address.
    xff = request.headers.get("x-forwarded-for", "")
    return xff.split(",")[0].strip() if xff else get_remote_address(request)


def _record(kind: str, request: Request, **fields):
    _ACTION_LOG.append({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "ip": _client_ip(request),
        **fields,
    })


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="The Construct API", version="1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# One comma-separated string. Override with the ALLOWED_ORIGINS env var to add
# preview URLs. Local dev needs a static server (Live Server etc.), not file://.
_DEFAULT_ORIGINS = ",".join([
    "https://construct.talcamusic.com",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "null",  # an index.html opened as a file:// sends Origin: null (local testing only)
])
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ActionRequest(BaseModel):
    user_action: str = Field(..., max_length=500, description="Player action string.")
    current_state: dict[str, Any]

class LocationOutput(BaseModel):
    location_name: str = Field(description="A moody, unique science fiction starting location name.")
    scenario_description: str = Field(description="An atmospheric starting scene description.")

@app.post("/api/turn", dependencies=[Depends(require_key)])
@limiter.limit("15/minute")
def play_turn(request: Request, body: ActionRequest):  # sync -> runs in a worker thread, never blocks the event loop
    try:
        game_state = body.current_state
        current_lang = game_state.get("language", "en")
        current_gender = game_state.get("player_gender", "Unspecified")
        turn_no = game_state.get("turn_count", 1)

        log.info(
            "TURN ip=%s lang=%s gender=%s turn=%s action=%r",
            get_remote_address(request), current_lang, current_gender,
            turn_no, body.user_action,
        )
        _record(
            "turn", request, lang=current_lang, gender=current_gender,
            turn=turn_no, action=body.user_action,
        )

        game_state.setdefault("narrative_history", [])
        game_state.setdefault("turn_log", [])
        game_state["narrative_history"].append(f"\nUSER: {body.user_action}\n")

        new_state = narrative_engine.invoke(game_state)

        if isinstance(new_state, dict):
            new_state["language"] = current_lang
            new_state["player_gender"] = current_gender

        print(f"[API DEBUG] image attached: {bool(new_state.get('latest_image_url'))}")
        return {"status": "success", "new_state": new_state}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-location", dependencies=[Depends(require_key)])
@limiter.limit("5/minute")
def generate_location(request: Request, body: dict):  # sync -> worker thread
    try:
        state = body.get("current_state", {})
        player_gender = state.get("player_gender", "Unspecified")
        player_lang = state.get("language", "en")

        log.info(
            "NEW GAME ip=%s lang=%s gender=%s",
            get_remote_address(request), player_lang, player_gender,
        )
        _record("new_game", request, lang=player_lang, gender=player_gender)

        lang_mapping = {'en': 'English', 'fr': 'French', 'de': 'German', 'ru': 'Russian'}
        target_language = lang_mapping.get(player_lang, 'English')

        system_prompt = f"""
        You are the Game Master of a high-stakes, wildly diverse science fiction interactive fiction.
        Generate a completely unique, surprising, and immersive starting location and scenario where the player wakes up or begins.
        Player Gender: {player_gender}
        CRITICAL: Write both `location_name` and `scenario_description` strictly in **{target_language}**.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=LocationOutput,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a completely randomized, non-repetitive sci-fi starting location in {target_language}."}
            ]
        )
        return {
            "status": "success",
            "location_name": response.location_name,
            "scenario_description": response.scenario_description
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feed")
@limiter.limit("60/minute")
def feed(request: Request, key: str = ""):
    _check_admin(key)
    return {"count": len(_ACTION_LOG), "entries": list(_ACTION_LOG)}


_ADMIN_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Construct :: live feed</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #0a0f0a; color: #7dff9b;
         font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  header { position: sticky; top: 0; background: #0a0f0a; border-bottom: 1px solid #1c3a24;
           padding: 10px 14px; display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap; }
  h1 { font-size: 14px; margin: 0; color: #9effb5; letter-spacing: 1px; }
  #meta { color: #4f8a63; }
  #err { color: #ff7d7d; }
  table { width: 100%; border-collapse: collapse; }
  td, th { padding: 4px 14px; text-align: left; vertical-align: top;
           border-bottom: 1px solid #12240f; white-space: nowrap; }
  td.action { white-space: pre-wrap; color: #eaffea; }
  tr.new_game td { color: #ffd27d; }
  th { position: sticky; top: 41px; background: #0a0f0a; color: #4f8a63; font-weight: normal; }
  .ip { color: #4f8a63; }
</style>
</head>
<body>
<header>
  <h1>THE CONSTRUCT :: LIVE FEED</h1>
  <span id="meta">connecting...</span>
  <span id="err"></span>
</header>
<table>
  <thead><tr><th>time (UTC)</th><th>ip</th><th>lang</th><th>gender</th><th>turn</th><th>action</th></tr></thead>
  <tbody id="rows"></tbody>
</table>
<script>
  const key = new URLSearchParams(location.search).get("key") || "";
  const rows = document.getElementById("rows");
  const meta = document.getElementById("meta");
  const err = document.getElementById("err");
  const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

  async function tick() {
    try {
      const r = await fetch(`/api/feed?key=${encodeURIComponent(key)}`, { cache: "no-store" });
      if (!r.ok) { err.textContent = `feed error ${r.status}` + (r.status === 401 ? " :: bad or missing ?key=" : ""); return; }
      err.textContent = "";
      const data = await r.json();
      const entries = data.entries.slice().reverse();  // newest first
      rows.innerHTML = entries.map(e => `
        <tr class="${esc(e.kind)}">
          <td>${esc(e.ts)}</td>
          <td class="ip">${esc(e.ip)}</td>
          <td>${esc(e.lang || "")}</td>
          <td>${esc(e.gender || "")}</td>
          <td>${e.kind === "new_game" ? "NEW" : esc(e.turn ?? "")}</td>
          <td class="action">${e.kind === "new_game" ? "(started a new game)" : esc(e.action || "")}</td>
        </tr>`).join("");
      meta.textContent = `${data.count} entries buffered - refreshed ${new Date().toLocaleTimeString()}`;
    } catch (e) {
      err.textContent = "network error";
    }
  }
  tick();
  setInterval(tick, 4000);
</script>
</body>
</html>"""


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return _ADMIN_HTML


@app.get("/")
async def root():
    return {"status": "online", "message": "The Construct is active."}
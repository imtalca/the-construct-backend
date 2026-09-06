from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from engine import narrative_engine, client

load_dotenv()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="The Construct API", version="1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# TODO: restrict allow_origins to the real frontend domain before going public
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.post("/api/turn")
@limiter.limit("15/minute")
async def play_turn(request: Request, body: ActionRequest):
    try:
        game_state = body.current_state
        current_lang = game_state.get("language", "en")
        current_gender = game_state.get("player_gender", "Unspecified")

        if "narrative_history" not in game_state:
            game_state["narrative_history"] = []
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

@app.post("/api/generate-location")
@limiter.limit("5/minute")
async def generate_location(request: Request, body: dict):
    try:
        state = body.get("current_state", {})
        player_gender = state.get("player_gender", "Unspecified")
        player_lang = state.get("language", "en")

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

@app.get("/")
async def root():
    return {"status": "online", "message": "The Construct is active."}
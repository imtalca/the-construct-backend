import os
from dotenv import load_dotenv
from openai import OpenAI
import instructor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any
from engine import narrative_engine, client

load_dotenv()

# Initialize the API
app = FastAPI(title="The Construct API", version="1.0")

# Allow your frontend to talk to this backend without security blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We can lock this down to your custom Netlify domain later if desired
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the exact JSON structure we expect from the frontend for turns
class ActionRequest(BaseModel):
    user_action: str
    current_state: dict[str, Any]

# Schema for random location generation output
class LocationOutput(BaseModel):
    location_name: str = Field(description="A moody, unique cyberpunk starting location name.")
    scenario_description: str = Field(description="An atmospheric starting scene description where the player wakes up or begins, fitting the target language and player gender.")

# Create the turn endpoint
@app.post("/api/turn")
async def play_turn(request: ActionRequest):
    try:
        # Extract the state and append the new user action
        game_state = request.current_state
        action_text = f"\nUSER: {request.user_action}\n"
        
        # If narrative_history doesn't exist yet, initialize it
        if "narrative_history" not in game_state:
            game_state["narrative_history"] = []
            
        game_state["narrative_history"].append(action_text)

        # Feed the state into your LangGraph engine
        new_state = narrative_engine.invoke(game_state)
        
        # Return the updated state to the frontend
        return {"status": "success", "new_state": new_state}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create the random location generation endpoint for the reroll loop
@app.post("/api/generate-location")
async def generate_location(request: dict):
    try:
        state = request.get("current_state", {})
        player_gender = state.get("player_gender", "Unspecified")
        player_lang = state.get("language", "en")
        
        lang_mapping = {'en': 'English', 'fr': 'French', 'de': 'German', 'ru': 'Russian'}
        target_language = lang_mapping.get(player_lang, 'English')
        
        system_prompt = f"""
        You are the Game Master of a gritty, high-stakes cyberpunk interactive fiction.
        Generate a unique, unexpected, and immersive starting location and scenario where the player wakes up or begins. 
        Avoid generic bio-labs unless twisted in a surreal way. Make it atmospheric and cinematic.
        Player Gender: {player_gender}
        CRITICAL: Write both `location_name` and `scenario_description` strictly in **{target_language}**.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=LocationOutput,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a random cyberpunk starting location in {target_language}."}
            ]
        )
        return {
            "status": "success",
            "location_name": response.location_name, 
            "scenario_description": response.scenario_description
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# A simple health check endpoint to verify the server is online
@app.get("/")
async def root():
    return {"status": "online", "message": "The Construct is active."}
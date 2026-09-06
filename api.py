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
        You are the Game Master of a high-stakes, deeply imaginative science fiction interactive fiction.
        Generate a radically unique, surprising, and immersive starting location and scenario where the player wakes up or begins. 
        
        RADICAL DIVERSITY & SCI-FI PALETTE (PULL FROM COMPLETELY DIFFERENT DOMAINS EACH TIME):
        - A hollowed-out comet core filled with suspended data-ghosts and zero-g magnetic fluid
        - Digital-industrial undertones, subtle reality-glitch anomalies, grid-locked data networks, and environments where physical space borders on terminal-driven architecture.
        - An atmospheric floating harvesting station suspended inside the crushing upper storms of a gas giant
        - A stellar-archive vault where memories and historical records are stored as pressurized optical gas columns
        - Towering retro-futuristic corporate monoliths, industrial decay, neon-lit urban sprawl, rain-slicked or synthetic atmosphere, high-tech low-life tension, and heavy noir shadows.
        - A derelict automated terraforming foundry choked with glowing magnetic particulate clouds
        - A deep-core tectonic pressure-rig extracting heavy exotic isotopes directly from a molten mantle
        - A macro-engineering Dyson ring maintenance strut vibrating under raw solar plasma pressure
        - Sleek yet socially tense municipal centers, cybernetic integration hubs, sterile corporate assembly lines, and sharp contrasts between high-end synthetic luxury and raw grit.

        ANTI-REPETITION & VARIETY RULES:
        - Actively avoid repeating the same themes or locking onto repetitive motifs. 
        - Rotate dynamically between the palette

            Make it atmospheric, uncanny, and cinematic. Ensure every generated location feels distinct, fresh, and varied.
        Player Gender: {player_gender}
        CRITICAL: Write both `location_name` and `scenario_description` strictly in **{target_language}**.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=LocationOutput,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a unique, highly varied sci-fi starting location blending cyberpunk, Blade Runner, Detroit: Become Human, and Matrix aesthetics in {target_language}."}
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
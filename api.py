from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
from engine import narrative_engine

# Initialize the API
app = FastAPI(title="The Construct API", version="1.0")

# Allow your future frontend to talk to this backend without security blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We will lock this down to your custom domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the exact JSON structure we expect from the frontend
class ActionRequest(BaseModel):
    user_action: str
    current_state: dict[str, Any]

# Create the endpoint
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

# A simple health check endpoint to verify the server is online
@app.get("/")
async def root():
    return {"status": "online", "message": "The Construct is active."}
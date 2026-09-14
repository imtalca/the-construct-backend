# THE CONSTRUCT

A terminal-style, AI-driven text adventure. You wake up inside a simulated
reality with no memory of how you got there — every choice you type is
evaluated live by an LLM narrative engine that reacts, escalates difficulty,
and eventually decides how (or whether) you get out.

**Play it live: [construct.talcamusic.com](https://construct.talcamusic.com/)**

> Access is gated by a key — reach out if you'd like one to try it.
> See [Gameplay Recording](#gameplay-recording) below for a full playthrough.

## How it works

- **Setup**: pick a language, a biometric classification, a psychometric
  profile (7 quick diagnostic questions that shape your starting stats), and
  now — a preferred number of turns (1–10) for the run.
- **Location**: the server generates a random atmospheric starting location;
  reroll until you're happy with it, then type your first action.
- **Play**: each turn you type a free-text action. The narrative engine
  evaluates it against your stats and the current difficulty, advances the
  story, and may hand you or take away inventory items.
- **Finale**: once you hit your chosen turn limit, the engine writes a short
  closing assessment based on everything you did.

## Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (`api.py`) — exposes
  `/api/turn` and `/api/generate-location`, rate-limited with `slowapi`, and
  a lightweight in-memory `/admin` live feed for monitoring player activity.
- **Narrative engine**: [LangGraph](https://github.com/langchain-ai/langgraph)
  (`engine.py`) — a small state graph (`game_master → router → finale`) that
  calls OpenAI through [`instructor`](https://github.com/jxnl/instructor) for
  structured, typed responses (success/failure, stat changes, inventory
  changes, image prompts) instead of parsing free-form text.
- **Frontend**: a single dependency-free `index.html` — a terminal-style UI
  driven by a phase state machine in vanilla JS, no build step required.
- **Images**: optional per-turn scene images via OpenAI's image API
  (`ENABLE_IMAGES=0` to disable for faster local testing).

Supports English, French, German, and Russian end-to-end (UI copy, profiler
questions, and LLM prompts).

## Deployment

- **Frontend**: static `index.html` deployed on [Netlify](https://www.netlify.com/).
- **Backend**: FastAPI app deployed on [Render](https://render.com/) (free
  tier — the instance spins down after ~15 min idle, so the first request
  after a while can take up to ~50s to wake it back up; the frontend retries
  location generation automatically to ride this out).

`API_BASE` in `index.html` points at the Render URL; `ALLOWED_ORIGINS` on the
backend must include the Netlify domain for CORS to allow it through.

## Project structure

```
api.py       FastAPI app: endpoints, rate limiting, request logging
engine.py    LangGraph narrative engine: prompts, difficulty, win condition
state.py     Shared EngineState schema passed through the graph
profiler.py  Psychometric profiler question/scoring support
index.html   Entire frontend: terminal UI + phase state machine
assets/      Background art, QR code, etc.
```

## Gameplay recording

A full playthrough is available in the
[v1.0-gameplay release](https://github.com/imtalca/the-construct-backend/releases/tag/v1.0-gameplay).

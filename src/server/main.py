import sys
import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI

# Adjust path to include src if running from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.server.api import app
from src.server.state import ServerState

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[Main] Server Starting...")
    state = ServerState.get_instance()
    try:
        state.start()
    except Exception as e:
        print(f"[Main] Failed to start state/camera: {e}")
        # We might continue to allow API to report error?

    yield

    # Shutdown
    print("[Main] Server Shutting Down...")
    state.stop()

# Assign lifespan to app (it wasn't set in api.py)
app.router.lifespan_context = lifespan

def main():
    # Run Uvicorn
    # host 0.0.0.0 to be accessible externally
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

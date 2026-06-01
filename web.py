"""
server.py — FastAPI WebSocket backend for FocusSense
Receives webcam frames from browser, runs AttentionEngine,
returns annotated frames + status back to browser.
"""

import cv2
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from main import AttentionEngine

app = FastAPI(title="FocusSense API")

# Load engine once at startup — not per request
engine = AttentionEngine()
engine.load()
print("[Server] AttentionEngine loaded and ready")

# Serve static files (index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")


import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

@app.get("/")
async def root():
    """Serve the main HTML page."""
    html_path = BASE_DIR / "static" / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint.
    Receives: base64 encoded JPEG frame from browser webcam
    Sends back: base64 encoded annotated frame + JSON status
    """
    await websocket.accept()
    print("[WebSocket] Client connected")

    try:
        while True:
            # Receive frame from browser
            data = await websocket.receive_text()

            # Decode base64 image
            img_data = base64.b64decode(data.split(",")[1])
            np_arr   = np.frombuffer(img_data, np.uint8)
            frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Run AttentionEngine inference
            state = engine.process(frame)

            # Encode annotated frame back to base64
            _, buffer  = cv2.imencode(".jpg", frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
            encoded    = base64.b64encode(buffer).decode("utf-8")
            img_base64 = f"data:image/jpeg;base64,{encoded}"

            # Build response
            response = {
                "frame"      : img_base64,
                "focused"    : state.get("focused", False),
                "gaze_label" : state.get("gaze_label", "—"),
                "gaze_conf"  : round(state.get("gaze_conf", 0.0), 2),
                "pose_label" : state.get("pose_label", "—"),
                "pose_conf"  : round(state.get("pose_conf", 0.0), 2),
                "df_label"   : state.get("df_label", "—"),
                "dist"       : round(state.get("dist", 0.0), 1),
            }

            import json
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        await websocket.close()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status"      : "running",
        "models_loaded": engine._models_ok
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
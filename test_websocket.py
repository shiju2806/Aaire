#!/usr/bin/env python3
"""
Test WebSocket connection to AAIRE
"""

import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/api/v1/chat/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            
            # Send a test message
            test_message = {
                "type": "query",
                "message": "what is capital in insurance",
                "session_id": "test-session"
            }
            
            print(f"📤 Sending: {test_message}")
            await websocket.send(json.dumps(test_message))
            
            # Wait for response
            print("⏳ Waiting for response...")
            response = await websocket.recv()
            print(f"📥 Received: {response}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
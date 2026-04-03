"""Simple OpenAI demo — wrap a client, make a call, see the ContextPack.

Usage:
    pip install glassbox-ctx[openai,server]
    export OPENAI_API_KEY=sk-...
    python examples/demo_openai.py
    # Then: glassbox serve
"""

import threading
from glassbox import wrap, serve
from openai import OpenAI

# Start viewer in background
threading.Thread(target=serve, daemon=True).start()

# Wrap the OpenAI client
client = wrap(OpenAI(), agent_name="demo-agent", app_name="glassbox-demo")

# Make a call — ContextPack is automatically captured
print("Making OpenAI call...")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=256,
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is context engineering and why does it matter for AI agents?"},
    ],
)

print(f"\nResponse: {response.choices[0].message.content[:200]}...")
print("\n  Open http://localhost:4100 to see the ContextPack")
print("  Press Ctrl+C to stop\n")

# Keep alive for viewer
try:
    threading.Event().wait()
except KeyboardInterrupt:
    pass

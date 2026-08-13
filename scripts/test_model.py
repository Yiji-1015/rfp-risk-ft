import os
from pathlib import Path
from dotenv import load_dotenv
import google.genai as genai

root_dir = Path(__file__).resolve().parent.parent
load_dotenv(root_dir / ".env")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("=== 사용 가능 모델 리스트 확인 ===")
try:
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            print(f"모델 ID: {m.name}")
except Exception as e:
    print(f"Error: {e}")

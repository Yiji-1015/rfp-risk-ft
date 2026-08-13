import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import google.genai as genai

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

root_dir = Path(__file__).resolve().parent.parent
load_dotenv(root_dir / ".env")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

test_models = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash-8b",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest"
]

print("=== 쿼터 여유 모델 탐색 테스트 ===")
for m in test_models:
    try:
        res = client.models.generate_content(
            model=m,
            contents="테스트입니다. 'OK'라고 정답만 말하세요."
        )
        print(f"[성공 ✅] 모델 '{m}' -> 정상 응답: {res.text.strip()}")
    except Exception as e:
        err = str(e)
        if "404" in err:
            print(f"[404 ❌] 모델 '{m}' -> 지원 안 됨")
        elif "429" in err:
            print(f"[429 ⚠️] 모델 '{m}' -> 쿼터 소진 (Resource Exhausted)")
        else:
            print(f"[오류 ❌] 모델 '{m}' -> {err[:100]}")

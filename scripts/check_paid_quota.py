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

print("=== 충전된 API 키 쿼터 및 고성능 모델 테스트 ===")
test_models = ["gemini-3.5-flash", "gemini-3.5-pro", "gemini-2.5-pro", "gemini-3.5-flash-lite"]

for m in test_models:
    try:
        res = client.models.generate_content(
            model=m,
            contents="유료 쿼터 확인 테스트입니다. '정상'이라고 답하세요."
        )
        print(f"[성공 🎉] 모델 '{m}' -> 응답: {res.text.strip()}")
    except Exception as e:
        print(f"[실패 ❌] 모델 '{m}' -> 에러: {e}")

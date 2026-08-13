import os
from pathlib import Path
from dotenv import load_dotenv
import google.genai as genai

root_dir = Path(__file__).resolve().parent.parent
load_dotenv(root_dir / ".env")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("=== 1건 호출 테스트 ===")
models_to_test = ["models/gemini-2.5-flash", "models/gemini-3.6-flash", "gemini-2.5-flash", "gemini-3.6-flash"]

for m in models_to_test:
    try:
        res = client.models.generate_content(
            model=m,
            contents="안녕하세요 테스트입니다. '안녕'이라고만 답변하세요."
        )
        print(f"[성공] 모델 '{m}' -> 응답: {res.text.strip()}")
        break
    except Exception as e:
        print(f"[실패] 모델 '{m}' -> 에러: {e}")

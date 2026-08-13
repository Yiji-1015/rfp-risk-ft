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

print("=== AI Studio Prepay 동기화 활성화 상태 체크 ===")
try:
    res = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="테스트입니다."
    )
    print(f"[동기화 완료 🎉] gemini-3.5-flash 성공: {res.text.strip()}")
except Exception as e:
    print(f"[동기화 진행 중/확인 필요 ⚠️] gemini-3.5-flash 에러: {e}")

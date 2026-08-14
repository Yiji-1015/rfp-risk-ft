"""Small Gemini API diagnostics.

These checks report only observable API results. They do not determine billing,
prepayment, or remaining quota status.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_MODEL = "gemini-2.5-flash"
ROOT_DIR = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List Gemini models or make one small smoke-test request.",
        epilog="Results do not establish quota, billing, or prepayment status.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "models",
        help="List models that advertise generateContent support.",
    )

    smoke = subparsers.add_parser(
        "smoke",
        help="Send one short generateContent request.",
    )
    smoke.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model ID (default: {DEFAULT_MODEL}).",
    )
    smoke.add_argument(
        "--prompt",
        default="Reply with OK.",
        help="Short prompt to send.",
    )
    return parser


def _load_api_key() -> str | None:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
    return os.getenv("GEMINI_API_KEY")


def _make_client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


def _list_models(client) -> int:
    for model in client.models.list():
        actions = model.supported_actions or []
        if "generateContent" in actions:
            print(model.name)
    return 0


def _smoke(client, model: str, prompt: str) -> int:
    response = client.models.generate_content(model=model, contents=prompt)
    print(response.text or "")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = _load_api_key()
    if not api_key:
        print(
            "GEMINI_API_KEY is missing. Add it to the environment or project .env file.",
            file=sys.stderr,
        )
        return 2

    try:
        client = _make_client(api_key)
        if args.command == "models":
            return _list_models(client)
        return _smoke(client, args.model, args.prompt)
    except Exception as exc:
        print(f"Gemini API request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

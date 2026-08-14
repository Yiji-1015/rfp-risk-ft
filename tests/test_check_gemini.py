"""Parser-only tests for the Gemini diagnostics CLI."""

from scripts.utilities.check_gemini import DEFAULT_MODEL, build_parser


def test_models_command_parses() -> None:
    args = build_parser().parse_args(["models"])

    assert args.command == "models"


def test_smoke_command_uses_safe_defaults() -> None:
    args = build_parser().parse_args(["smoke"])

    assert args.command == "smoke"
    assert args.model == DEFAULT_MODEL
    assert args.prompt == "Reply with OK."


def test_smoke_command_accepts_model_and_prompt() -> None:
    args = build_parser().parse_args(
        ["smoke", "--model", "gemini-test", "--prompt", "hello"]
    )

    assert args.model == "gemini-test"
    assert args.prompt == "hello"

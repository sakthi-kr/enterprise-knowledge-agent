"""Command-line entry point for grounded RAG questions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from enterprise_knowledge_agent.runtime import get_answer_service


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Ask a grounded enterprise knowledge question.")
    parser.add_argument("question")
    return parser


def main() -> None:
    """Run one grounded enterprise question and print structured JSON."""

    args = build_parser().parse_args()
    result = get_answer_service().answer(args.question)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

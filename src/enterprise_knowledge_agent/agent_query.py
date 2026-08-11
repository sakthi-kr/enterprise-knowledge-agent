"""Run one enterprise knowledge agent query from the command line."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from enterprise_knowledge_agent.runtime import get_agent_service


def main() -> None:
    """Execute one agent query and print structured JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    args = parser.parse_args()

    result = get_agent_service().run(args.question)
    payload = asdict(result)
    payload["plan"]["strategy"] = result.plan.strategy.value
    payload["answer"]["status"] = result.answer.status.value
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

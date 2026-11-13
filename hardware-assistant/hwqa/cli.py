"""Command line interface.

    hwqa                          interactive chat (auto-selects a local Ollama model, else rule mode)
    hwqa ask "Is my SSD NVMe?"    one question
    hwqa tools                    what can be read on this machine
    hwqa dump                     every tool's structured output as JSON (no model)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import Optional

from .assistant import Assistant
from .backends import BackendError, OllamaBackend, OpenAICompatBackend, RuleBackend
from .system import FixtureSystem, LiveSystem
from .tools import TOOLS

BANNER = """hwqa - ask about this machine's hardware. Type /help for commands, /quit to exit.
backend: {backend}{privacy}"""

HELP = """/tools    what each tool can read on this machine
/sources  commands and files behind the last answer
/facts    raw structured data from the last answer
/reset    forget the conversation (cached hardware data is kept)
/quit     exit"""


def build_system(args):
    if args.fixtures:
        return FixtureSystem(args.fixtures, sudo_allowed=args.sudo)
    if platform.system() != "Linux":
        sys.exit("hwqa reads Linux /proc, /sys and util-linux tools. On other systems use --fixtures "
                 "fixtures/thinkpad-e14 to try it with a recorded laptop.")
    return LiveSystem(sudo_allowed=args.sudo)


def build_backend(args):
    if args.backend == "rules":
        return RuleBackend(), ""
    try:
        if args.backend == "openai":
            backend = OpenAICompatBackend(model=args.model or os.environ.get("HWQA_MODEL", "local-model"),
                                          base_url=args.base_url or os.environ.get("HWQA_BASE_URL", "http://localhost:8080/v1"),
                                          api_key=os.environ.get("HWQA_API_KEY", ""), allow_remote=args.allow_remote)
            return backend, ""
        backend = OllamaBackend(model=args.model or os.environ.get("HWQA_MODEL", "qwen2.5:7b-instruct"),
                                base_url=args.base_url or os.environ.get("HWQA_OLLAMA_URL", "http://localhost:11434"),
                                allow_remote=args.allow_remote)
    except BackendError as exc:
        sys.exit(str(exc))
    if backend.available():
        return backend, ""
    if args.backend == "ollama":
        sys.exit(f"Ollama is not reachable or model '{backend.model}' is not pulled. Run: ollama pull {backend.model}")
    return RuleBackend(), f"  (no local model found at {backend.client.base_url}; using rule mode)"


def describe(backend) -> str:
    if isinstance(backend, RuleBackend):
        return "rules (no LLM, fully offline)"
    where = "local - hardware data stays on this machine" if backend.local else "REMOTE - hardware data is sent off this machine"
    return f"{backend.name} / {backend.model} ({where})"


def print_answer(ans, verbose: bool) -> None:
    if verbose and ans.tool_calls:
        calls = ", ".join(c["name"] + (" (cached)" if c["cached"] else "") + (f" ERROR {c['error']}" if c["error"] else "")
                          for c in ans.tool_calls)
        print(f"  [tools: {calls}]")
    print(ans.text)


def repl(assistant: Assistant, verbose: bool, banner: str) -> None:
    try:
        import readline  # noqa: F401  (line editing + history when available)
    except ImportError:
        pass
    print(banner)
    last = None
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        if q in ("/quit", "/exit"):
            return
        if q == "/help":
            print(HELP)
        elif q == "/tools":
            for name, status in assistant.capabilities.items():
                print(f"  {name:<22} {status}")
        elif q == "/sources":
            print("\n".join(f"  {s}" for s in (last.sources if last else [])) or "  (none yet)")
        elif q == "/facts":
            print(json.dumps(assistant.session_results[-len(last.tool_calls):] if last and last.tool_calls else [], indent=2))
        elif q == "/reset":
            assistant.reset()
            print("  conversation cleared")
        else:
            try:
                last = assistant.ask(q)
            except BackendError as exc:
                print(f"  model error: {exc}")
                continue
            print("hwqa> ", end="")
            print_answer(last, verbose)


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(prog="hwqa", description="Ask questions about this Linux machine's hardware.")
    ap.add_argument("command", nargs="?", default="chat", choices=["chat", "ask", "tools", "dump"])
    ap.add_argument("question", nargs="*")
    ap.add_argument("--backend", choices=["auto", "ollama", "openai", "rules"], default="auto")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--allow-remote", action="store_true", help="allow a model server that is not on this machine")
    ap.add_argument("--sudo", action="store_true",
                    help="use `sudo -n` for root-only tools (needs contrib/sudoers.d/hwqa installed)")
    ap.add_argument("--fixtures", help="replay a recorded machine instead of reading this one")
    ap.add_argument("-v", "--verbose", action="store_true", help="show which tools were called")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_intermixed_args(argv)

    system = build_system(args)
    if args.command == "tools":
        a = Assistant(system)
        for name, status in a.capabilities.items():
            print(f"{name:<22} {status:<40} {TOOLS[name].description[:60]}")
        return
    if args.command == "dump":
        print(json.dumps(Assistant(system).collect_all(), indent=2, ensure_ascii=False))
        return

    backend, note = build_backend(args)
    assistant = Assistant(system, backend)
    if args.command == "ask":
        if not args.question:
            ap.error("ask needs a question")
        ans = assistant.ask(" ".join(args.question))
        if args.json:
            print(json.dumps({"answer": ans.text, "tool_calls": ans.tool_calls, "sources": ans.sources,
                              "unverified": ans.unverified}, indent=2, ensure_ascii=False))
        else:
            print_answer(ans, args.verbose)
        return
    privacy = "\n" + note if note else ""
    repl(assistant, args.verbose, BANNER.format(backend=describe(backend), privacy=privacy))


if __name__ == "__main__":
    main()

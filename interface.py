"""CLI interface for the chatbot.

Run with:
  python interface.py --remote   # Use HF router (requires HF_TOKEN)
  python interface.py            # Try remote then fallback to local

"""
from __future__ import annotations
import argparse
import sys
from typing import List, Dict

from model_loader import get_chat_backend, ModelPendingDeployError
import os
from memory import SlidingWindowMemory

EXIT_COMMAND = "/exit"
RESET_COMMAND = "/reset"
HELP_COMMAND = "/help"

HELP_TEXT = f"""Commands:
  {EXIT_COMMAND}  - exit the chatbot
  {RESET_COMMAND} - clear conversation memory
  {HELP_COMMAND}  - show this help
"""

def build_arg_parser():
    p = argparse.ArgumentParser(description="Local / remote Hugging Face CLI chatbot")
    p.add_argument("--remote", action="store_true", help="Force remote (HF router) mode.")
    p.add_argument("--local-only", action="store_true", help="Skip remote attempt and use local pipeline directly.")
    p.add_argument("--remote-only", action="store_true", help="Do not fallback to local if remote fails (error instead).")
    p.add_argument("--remote-model", type=str, default=None, help="Override remote model id.")
    p.add_argument("--local-model", type=str, default=None, help="Override local model id.")
    p.add_argument("--window", type=int, default=5, help="Memory window (turns).")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    p.add_argument("--stream", action="store_true", help="Stream tokens for remote backend (if supported).")
    return p


def main():
    args = build_arg_parser().parse_args()

    memory = SlidingWindowMemory(max_turns=args.window)
    prefer_remote = not args.local_only
    if args.remote:
        prefer_remote = True
    elif args.local_only:
        prefer_remote = False

    backend = get_chat_backend(prefer_remote=prefer_remote,
                               remote_model=args.remote_model,
                               local_model=args.local_model,
                               allow_fallback=not args.remote_only)
    mode = "remote" if prefer_remote else "local"
    token_status = "present" if os.getenv("HF_TOKEN") else "missing"
    fallback_status = "enabled" if not args.remote_only else "disabled"
    print(f"[Startup] Mode preference: {mode} | HF_TOKEN: {token_status} | Fallback: {fallback_status}")
    print("Chatbot started. Type /help for commands.\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chatbot. Goodbye!")
            return

        if not user_input:
            continue
        if user_input.lower() == EXIT_COMMAND:
            print("Exiting chatbot. Goodbye!")
            return
        if user_input.lower() == HELP_COMMAND:
            print(HELP_TEXT)
            continue
        if user_input.lower() == RESET_COMMAND:
            memory.clear()
            print("[Memory cleared]")
            continue

        memory.add("user", user_input)
        messages: List[Dict[str, str]] = memory.get()

        try:
            if args.stream and hasattr(backend, 'generate'):
                try:
                    reply = backend.generate(messages, stream=True)  # type: ignore
                except TypeError:
                    reply = backend.generate(messages)  # type: ignore
            else:
                reply = backend.generate(messages)  # type: ignore
        except ModelPendingDeployError as warm_e:
            if args.remote_only:
                reply = f"[Error] Remote model still warming: {warm_e}. Try again shortly." 
            else:
                print("[Info] Remote model warming too long; switching to local fallback.")
                backend = get_chat_backend(prefer_remote=False, local_model=args.local_model)
                try:
                    reply = backend.generate(messages)  # type: ignore
                except Exception as final_e:
                    reply = f"[Error during local fallback generation: {final_e}]"
        except Exception as e:
            reply = f"[Error during generation: {e}]"

        print(f"Bot: {reply}")
        memory.add("assistant", reply)


if __name__ == "__main__":
    main()

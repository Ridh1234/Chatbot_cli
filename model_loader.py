"""Model loading utilities for the CLI chatbot.

Uses Hugging Face Inference Endpoint (router) via OpenAI-compatible client
OR falls back to a local small model through transformers pipeline if desired.

Environment Variables:
- HF_TOKEN: your Hugging Face token (do NOT hardcode)      
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # Provide a friendly message if openai isn't installed
    OpenAI = None  # type: ignore

# Try to load .env automatically (supports user's provided HF_TOKEN in .env)
def _load_dotenv_if_present():
    loaded = False
    try:
        from dotenv import load_dotenv  # type: ignore
        if load_dotenv():
            loaded = True
    except Exception:
        # Fallback minimal parser if python-dotenv not installed
        env_path = os.path.join(os.getcwd(), '.env')
        if os.path.isfile(env_path):
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        key, val = line.split('=', 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = val
                            loaded = True
            except Exception:
                pass
    return loaded

_load_dotenv_if_present()

try:
    from transformers import pipeline  # type: ignore
except ImportError:
    pipeline = None  # type: ignore

HF_ROUTER_BASE = "https://router.huggingface.co/v1"
DEFAULT_REMOTE_MODEL = "google/gemma-3-27b-it:featherless-ai"
# Ordered list of fallback remote chat-capable models (smaller first for faster warm load)
REMOTE_MODEL_CANDIDATES = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mistral-7B-Instruct-v0.3",
    DEFAULT_REMOTE_MODEL,  # keep requested one in list too
]
class ModelPendingDeployError(RuntimeError):
    """Raised when the remote model remains in warming state after retries."""
    pass
DEFAULT_LOCAL_MODEL = "distilgpt2"  # Fallback tiny local model


@dataclass
class RemoteChatClient:
    client: Any
    model: str
    announced_ready: bool = False

    def generate(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: float = 0.7,
        retries: int = 5,
        initial_backoff: float = 3.0,
    ) -> str:
        """Generate with retry/backoff while model warms up.

        Retries when server response indicates model warming (e.g., model_pending_deploy).
        """
        backoff = initial_backoff
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=stream,
                    temperature=temperature,
                )
                if stream:
                    chunks = []
                    for chunk in completion:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            print(delta, end="", flush=True)
                            chunks.append(delta)
                    print()
                    if attempt > 1 and not self.announced_ready:
                        print("[Remote] Model is now ready.")
                        self.announced_ready = True
                    return "".join(chunks)
                else:
                    if attempt > 1 and not self.announced_ready:
                        print("[Remote] Model is now ready.")
                        self.announced_ready = True
                    return completion.choices[0].message.content  # type: ignore
            except Exception as e:  # Inspect message for warm-up indicators
                err_txt = str(e).lower()
                last_err = e
                if ("model_pending_deploy" in err_txt or "not ready for inference" in err_txt) and attempt < retries:
                    print(f"[Remote] Model warming (attempt {attempt}/{retries}). Retry in {backoff:.1f}s...")
                    time.sleep(backoff)
                    backoff *= 1.6
                    continue
                break
        if last_err:
            txt = str(last_err).lower()
            if "model_pending_deploy" in txt or "not ready for inference" in txt:
                raise ModelPendingDeployError(str(last_err))
            raise last_err
        raise RuntimeError("Unknown remote generation failure without captured exception.")


def load_remote_client(model: str = DEFAULT_REMOTE_MODEL) -> RemoteChatClient:
    """Instantiate an OpenAI-compatible client pointed at Hugging Face router.

    Environment override: HF_REMOTE_MODEL can set/override model id.
    """
    env_model = os.getenv("HF_REMOTE_MODEL")
    if env_model:
        model = env_model
    if OpenAI is None:
        raise RuntimeError("The 'openai' package is required for remote inference. Install via 'pip install openai'.")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable not set. Please export / set it before running.")
    client = OpenAI(base_url=HF_ROUTER_BASE, api_key=hf_token)
    return RemoteChatClient(client=client, model=model)


@dataclass
class LocalPipeline:
    generator: Any
    system_prompt: str = "You are a helpful assistant."

    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        convo = [f"System: {self.system_prompt}"]
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role == "user":
                convo.append(f"User: {content}")
            elif role == "assistant":
                convo.append(f"Assistant: {content}")
        convo.append("Assistant:")
        return "\n".join(convo)

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.8, max_new_tokens: int = 256) -> str:
        prompt = self.build_prompt(messages)
        out = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        text = out[0]["generated_text"]
        # Return only assistant's last part after the final 'Assistant:' marker
        if "Assistant:" in text:
            return text.split("Assistant:")[-1].strip()
        return text.strip()


def load_local_pipeline(model_name: str = DEFAULT_LOCAL_MODEL) -> LocalPipeline:
    if pipeline is None:
        raise RuntimeError("The 'transformers' package is required for local mode. Install via 'pip install transformers'.")
    generator = pipeline("text-generation", model=model_name)
    return LocalPipeline(generator=generator)


def get_chat_backend(prefer_remote: bool = True, remote_model: Optional[str] = None, local_model: Optional[str] = None, allow_fallback: bool = True):
    """Return a backend object with a unified 'generate(messages, **kwargs)' interface.

    If allow_fallback=False and remote load fails, the exception is raised to caller.
    """
    if prefer_remote:
        try:
            chosen = remote_model or DEFAULT_REMOTE_MODEL
            return load_remote_client(model=chosen)
        except Exception as e:
            if allow_fallback:
                print(f"[WARN] Remote client load failed: {e}. Falling back to local pipeline...")
            else:
                raise
    return load_local_pipeline(model_name=local_model or DEFAULT_LOCAL_MODEL)


def iterate_remote_backends(explicit_model: Optional[str] = None):
    """Yield RemoteChatClient objects for candidate models (fastest first)."""
    seen = set()
    order = []
    if explicit_model:
        order.append(explicit_model)
    order.extend(REMOTE_MODEL_CANDIDATES)
    for m in order:
        if m in seen:
            continue
        seen.add(m)
        try:
            yield load_remote_client(model=m)
        except Exception as e:
            print(f"[Remote-Fallback] Skipping model '{m}': {e}")
            continue

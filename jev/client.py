"""Thin HTTP client for llama-server.

Only standard llama.cpp endpoints are used:
  * /health, /props
  * /tokenize, /detokenize        (token pinning + boundary validation)
  * /apply-template               (chat-template parity, thinking control)
  * /completion                   (one forward pass, grammar + probabilities)
"""

from __future__ import annotations

from typing import Any

import httpx


class LlamaClientError(RuntimeError):
    pass


def _raw_base64(payload: str) -> str:
    """Strip a data URL prefix if present; /completion wants raw base64."""
    if payload.startswith("data:") and "," in payload:
        return payload.split(",", 1)[1]
    return payload


class LlamaClient:
    def __init__(self, base_url: str, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    def close(self) -> None:
        self._http.close()

    def _post(self, path: str, payload: dict) -> dict:
        try:
            r = self._http.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise LlamaClientError(f"{path} request failed: {exc}") from exc
        if r.status_code != 200:
            raise LlamaClientError(f"{path} -> HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def health(self) -> dict:
        r = self._http.get("/health")
        return r.json()

    def props(self) -> dict:
        return self._http.get("/props").json()

    # ------------------------------------------------------------------
    def tokenize(
        self,
        content: str,
        add_special: bool = False,
        parse_special: bool = True,
        with_pieces: bool = False,
    ) -> list:
        data = self._post(
            "/tokenize",
            {
                "content": content,
                "add_special": add_special,
                "parse_special": parse_special,
                "with_pieces": with_pieces,
            },
        )
        return data["tokens"]

    def detokenize(self, tokens: list[int]) -> str:
        return self._post("/detokenize", {"tokens": tokens})["content"]

    def apply_template(
        self,
        messages: list[dict],
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        payload: dict[str, Any] = {
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
        }
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        return self._post("/apply-template", payload)["prompt"]

    # ------------------------------------------------------------------
    def completion(
        self,
        prompt: str,
        *,
        grammar: str | None = None,
        n_predict: int = 1,
        n_probs: int = 16,
        post_sampling_probs: bool = True,
        multimodal_data: list[str] | None = None,
        cache_prompt: bool = True,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        seed: int = -1,
    ) -> dict:
        prompt_field: Any
        if multimodal_data:
            # /completion expects raw base64 payloads, not data URLs.
            prompt_field = {
                "prompt_string": prompt,
                "multimodal_data": [_raw_base64(x) for x in multimodal_data],
            }
        else:
            prompt_field = prompt

        payload: dict[str, Any] = {
            "prompt": prompt_field,
            "n_predict": n_predict,
            "n_probs": n_probs,
            "post_sampling_probs": post_sampling_probs,
            "cache_prompt": cache_prompt,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            "repeat_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "seed": seed,
            "stream": False,
        }
        if grammar:
            payload["grammar"] = grammar
        return self._post("/completion", payload)

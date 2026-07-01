"""vLLM generation backend (V8.2 generation/vllm)."""

from src.runners.vllm_runner import build_llm, generate_chunk, render_prompt

__all__ = ["build_llm", "generate_chunk", "render_prompt"]

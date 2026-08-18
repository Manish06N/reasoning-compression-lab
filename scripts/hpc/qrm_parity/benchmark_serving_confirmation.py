#!/usr/bin/env python3
"""Measured serving systems confirmation benchmark for Reasoning LLMs on NVIDIA A100.

Implements strict apples-to-apples controls:
1. Balanced Condition A: 20 MATH-500 problems (4 per level 1-5).
2. Pinned Condition B: 100 MATH-500 problems with explicit max_num_seqs=8 pinned & verified in vLLM.
3. Secondary Microbenchmark: Fixed 512 tokens pure decoding speed.
4. Technical Timing Replicates: Shared sampling seed (20260816) with CV-governed replicate sufficiency.
5. VRAM distinction: Explicitly separates model-weights memory from total engine pool (gpu_memory_utilization=0.75).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[3]


def get_gpu_info() -> Dict[str, Any]:
    """Capture runtime GPU environment and driver metadata."""
    info: Dict[str, Any] = {
        "hostname": platform.node(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else "unknown",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpu_devices"].append({
                "device_id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": props.total_memory / (1024**3),
                "multi_processor_count": getattr(props, "multi_processor_count", None),
            })
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,name,uuid,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        info["nvidia_smi"] = smi_out.strip()
    except Exception as e:
        info["nvidia_smi_error"] = str(e)
    return info


def run_condition_a_single_stream(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    repetition: int,
    model_name: str,
    weight_format: str,
    raw_dir: Path,
    gpu_meta: Dict[str, Any],
    weights_memory_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """Condition A: Sequential single-request stream (concurrency=1, 20 prompts, 4 per level)."""
    run_file = raw_dir / f"{model_name}_{weight_format}_rep{repetition}_condA.json"
    if run_file.exists():
        try:
            cached = json.loads(run_file.read_text(encoding="utf-8"))
            if cached.get("output_tokens_per_second", 0) > 0 and cached.get("n_requests", 0) == len(prompts):
                print(
                    f"  [Cond A Rep {repetition}] (Cached) Output tok/s: {cached['output_tokens_per_second']:.2f} | Latency median: {cached['latency_median_sec']:.2f}s | Peak VRAM: {cached['peak_vram_allocated_gb']:.2f}GB",
                    flush=True,
                )
                return cached
        except Exception:
            pass

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies: List[float] = []
    output_token_counts: List[int] = []
    input_token_counts: List[int] = []

    t_start = time.perf_counter()
    for idx, p in enumerate(prompts):
        req_start = time.perf_counter()
        outputs = llm.generate([p], sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        req_end = time.perf_counter()

        req_latency = req_end - req_start
        latencies.append(req_latency)

        out = outputs[0]
        in_toks = len(out.prompt_token_ids) if out.prompt_token_ids else 0
        out_toks = len(out.outputs[0].token_ids) if out.outputs else 0

        input_token_counts.append(in_toks)
        output_token_counts.append(out_toks)

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    total_out_toks = sum(output_token_counts)
    total_in_toks = sum(input_token_counts)
    n_reqs = len(prompts)

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)

    out_tok_per_sec = total_out_toks / elapsed if elapsed > 0 else 0.0
    req_per_sec = n_reqs / elapsed if elapsed > 0 else 0.0
    gpu_sec_per_query = elapsed / n_reqs if n_reqs > 0 else 0.0

    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p90_lat = float(np.percentile(latencies, 90))
    p95_lat = float(np.percentile(latencies, 95))

    result = {
        "benchmark_type": "task_realistic_confirmation",
        "condition": "A_single_stream_c1",
        "concurrency": 1,
        "model": model_name,
        "format": weight_format,
        "repetition": repetition,
        "n_requests": n_reqs,
        "level_stratification": "4_per_level_1_to_5",
        "total_input_tokens": total_in_toks,
        "total_output_tokens": total_out_toks,
        "mean_output_tokens_per_req": total_out_toks / n_reqs if n_reqs > 0 else 0.0,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": out_tok_per_sec,
        "requests_per_second": req_per_sec,
        "gpu_seconds_per_query": gpu_sec_per_query,
        "latency_mean_sec": mean_lat,
        "latency_median_sec": median_lat,
        "latency_p90_sec": p90_lat,
        "latency_p95_sec": p95_lat,
        "model_weights_memory_gb": weights_memory_gb,
        "peak_vram_allocated_gb": peak_vram_gb,
        "peak_vram_reserved_gb": peak_reserved_gb,
        "gpu_metadata": gpu_meta,
    }

    run_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"  [Cond A Rep {repetition}] Output tok/s: {out_tok_per_sec:.2f} | Latency median: {median_lat:.2f}s | P90: {p90_lat:.2f}s | Peak VRAM: {peak_vram_gb:.2f}GB",
        flush=True,
    )
    return result


def run_condition_b_batched_throughput(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    repetition: int,
    model_name: str,
    weight_format: str,
    raw_dir: Path,
    gpu_meta: Dict[str, Any],
    weights_memory_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """Condition B: Batched continuous batching with pinned max_num_seqs=8 (100 prompts, 20 per level)."""
    run_file = raw_dir / f"{model_name}_{weight_format}_rep{repetition}_condB.json"
    if run_file.exists():
        try:
            cached = json.loads(run_file.read_text(encoding="utf-8"))
            if (
                cached.get("output_tokens_per_second", 0) > 0
                and cached.get("n_requests", 0) == len(prompts)
                and cached.get("active_max_num_seqs") == 8
            ):
                print(
                    f"  [Cond B Rep {repetition}] (Cached) Output tok/s: {cached['output_tokens_per_second']:.2f} | Req/s: {cached['requests_per_second']:.3f} | GPU-sec/q: {cached['gpu_seconds_per_query']:.2f}s | Peak VRAM: {cached['peak_vram_allocated_gb']:.2f}GB",
                    flush=True,
                )
                return cached
        except Exception:
            pass

    # Verify scheduler config has max_num_seqs=8
    scheduler_max_seqs = llm.llm_engine.scheduler_config.max_num_seqs
    assert scheduler_max_seqs == 8, f"Engine scheduler max_num_seqs is {scheduler_max_seqs}, expected 8!"

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    n_reqs = len(prompts)

    input_tokens = [len(out.prompt_token_ids) if out.prompt_token_ids else 0 for out in outputs]
    output_tokens = [len(out.outputs[0].token_ids) if out.outputs else 0 for out in outputs]

    total_out_toks = sum(output_tokens)
    total_in_toks = sum(input_tokens)

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)

    out_tok_per_sec = total_out_toks / elapsed if elapsed > 0 else 0.0
    req_per_sec = n_reqs / elapsed if elapsed > 0 else 0.0
    gpu_sec_per_query = elapsed / n_reqs if n_reqs > 0 else 0.0

    result = {
        "benchmark_type": "task_realistic_confirmation",
        "condition": "B_batched_throughput_c8",
        "concurrency": 8,
        "active_max_num_seqs": scheduler_max_seqs,
        "max_num_seqs_confirmed": True,
        "model": model_name,
        "format": weight_format,
        "repetition": repetition,
        "n_requests": n_reqs,
        "level_stratification": "20_per_level_1_to_5",
        "total_input_tokens": total_in_toks,
        "total_output_tokens": total_out_toks,
        "mean_output_tokens_per_req": total_out_toks / n_reqs if n_reqs > 0 else 0.0,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": out_tok_per_sec,
        "requests_per_second": req_per_sec,
        "gpu_seconds_per_query": gpu_sec_per_query,
        "model_weights_memory_gb": weights_memory_gb,
        "peak_vram_allocated_gb": peak_vram_gb,
        "peak_vram_reserved_gb": peak_reserved_gb,
        "gpu_metadata": gpu_meta,
    }

    run_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"  [Cond B Rep {repetition}] Output tok/s: {out_tok_per_sec:.2f} | Req/s: {req_per_sec:.3f} | GPU-sec/q: {gpu_sec_per_query:.2f}s | Peak VRAM: {peak_vram_gb:.2f}GB (max_num_seqs=8 verified)",
        flush=True,
    )
    return result


def run_fixed_token_microbenchmark(
    llm: LLM,
    sample_prompts: List[str],
    model_name: str,
    weight_format: str,
    raw_dir: Path,
    gpu_meta: Dict[str, Any],
    weights_memory_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """Secondary microbenchmark: Fixed generated tokens (512 tokens) to isolate raw decoding speed."""
    run_file = raw_dir / f"{model_name}_{weight_format}_microbenchmark.json"
    if run_file.exists():
        try:
            cached = json.loads(run_file.read_text(encoding="utf-8"))
            if cached.get("raw_decode_tokens_per_second", 0) > 0:
                print(
                    f"  [Microbenchmark] (Cached) Raw decode tok/s (fixed 512 tokens): {cached['raw_decode_tokens_per_second']:.2f}",
                    flush=True,
                )
                return cached
        except Exception:
            pass

    fixed_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        min_tokens=512,
        ignore_eos=True,
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    outputs = llm.generate(sample_prompts[:10], fixed_params, use_tqdm=False)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    raw_tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

    result = {
        "benchmark_type": "fixed_token_microbenchmark_confirmation",
        "model": model_name,
        "format": weight_format,
        "n_requests": len(sample_prompts[:10]),
        "fixed_tokens_per_request": 512,
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "raw_decode_tokens_per_second": raw_tok_per_sec,
        "model_weights_memory_gb": weights_memory_gb,
        "gpu_metadata": gpu_meta,
    }

    run_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"  [Microbenchmark] Raw decode tok/s (fixed 512 tokens): {raw_tok_per_sec:.2f}",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Measured serving confirmation benchmark with strict controls.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to checkpoint directory.")
    parser.add_argument("--model-name", type=str, required=True, choices=["Qwen-7B", "Llama-8B"])
    parser.add_argument("--format", type=str, required=True, choices=["BF16", "FP8", "AWQ-4", "GPTQ-4"])
    parser.add_argument(
        "--cond-a-subset",
        type=Path,
        default=REPO_ROOT / "results" / "measured_serving_confirmation" / "condition_a_subset.json",
    )
    parser.add_argument(
        "--cond-b-subset",
        type=Path,
        default=REPO_ROOT / "results" / "measured_serving_confirmation" / "condition_b_subset.json",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "results" / "measured_serving_confirmation" / "raw",
    )
    parser.add_argument(
        "--provenance-dir",
        type=Path,
        default=REPO_ROOT / "results" / "reports" / "measured_serving_confirmation" / "provenance",
    )
    parser.add_argument("--min-repetitions", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.provenance_dir.mkdir(parents=True, exist_ok=True)

    gpu_meta = get_gpu_info()
    prov_file = args.provenance_dir / f"{gpu_meta['hostname']}_node_provenance.json"
    prov_file.write_text(json.dumps(gpu_meta, indent=2), encoding="utf-8")

    print("=" * 80, flush=True)
    print(f"CONFIRMATION SERVING BENCHMARK: {args.model_name} {args.format}", flush=True)
    print(f"Host: {gpu_meta['hostname']} | GPU: {gpu_meta.get('gpu_devices', [{}])[0].get('name', 'Unknown')}", flush=True)
    print(f"Checkpoint: {args.model_path}", flush=True)
    print("=" * 80, flush=True)

    # Check if all minimum runs already exist
    all_exist = True
    for rep in range(1, args.min_repetitions + 1):
        f_a = args.raw_dir / f"{args.model_name}_{args.format}_rep{rep}_condA.json"
        f_b = args.raw_dir / f"{args.model_name}_{args.format}_rep{rep}_condB.json"
        if not (f_a.exists() and f_b.exists()):
            all_exist = False
            break
    f_micro = args.raw_dir / f"{args.model_name}_{args.format}_microbenchmark.json"
    if not f_micro.exists():
        all_exist = False

    if all_exist:
        # Check CV of existing runs
        b_toks = [
            json.loads((args.raw_dir / f"{args.model_name}_{args.format}_rep{r}_condB.json").read_text())["output_tokens_per_second"]
            for r in range(1, args.min_repetitions + 1)
        ]
        mean_b = float(np.mean(b_toks))
        std_b = float(np.std(b_toks))
        cv_b = (std_b / mean_b * 100.0) if mean_b > 0 else 0.0
        if cv_b <= 3.0:
            print(f"All benchmarks for {args.model_name} {args.format} already completed (CV={cv_b:.2f}% <= 3%). Skipping execution.", flush=True)
            return

    # Determine dtype
    if args.format in ["AWQ-4", "GPTQ-4"]:
        dtype_str = "float16"
    else:
        dtype_str = "bfloat16"

    # Load input prompts
    cond_a_data = json.loads(args.cond_a_subset.read_text(encoding="utf-8"))
    prompts_a = [item["full_prompt"] for item in cond_a_data]
    print(f"Loaded {len(prompts_a)} Condition A balanced prompts (4 per level 1-5).", flush=True)

    cond_b_data = json.loads(args.cond_b_subset.read_text(encoding="utf-8"))
    prompts_b = [item["full_prompt"] for item in cond_b_data]
    print(f"Loaded {len(prompts_b)} Condition B balanced prompts (20 per level 1-5).", flush=True)

    # Initialize vLLM with max_num_seqs=8 pinned
    print(f"Initializing vLLM 0.7.0 (dtype={dtype_str}, max_len={args.max_model_len}, max_num_seqs=8, eager=True)...", flush=True)
    llm = LLM(
        model=str(args.model_path),
        dtype=dtype_str,
        max_model_len=args.max_model_len,
        max_num_seqs=8,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        tensor_parallel_size=1,
        seed=20260816,
    )

    # Assert active max_num_seqs
    active_max_seqs = llm.llm_engine.scheduler_config.max_num_seqs
    print(f"Verified engine scheduler max_num_seqs: {active_max_seqs}", flush=True)
    assert active_max_seqs == 8, f"Engine scheduler max_num_seqs is {active_max_seqs}, expected 8!"

    weights_memory_gb = torch.cuda.memory_allocated() / (1024**3)
    print(f"Model loaded successfully. Weights memory: {weights_memory_gb:.2f} GB", flush=True)

    # Warmup
    print("Executing 3 warmup requests (excluded from measurements)...", flush=True)
    warmup_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=128)
    warmup_prompts = [
        "<｜begin▁of▁sentence｜><｜User｜>What is 2 + 2?<｜Assistant｜><think>\n",
        "<｜begin▁of▁sentence｜><｜User｜>Simplify x^2 - 4 = 0.<｜Assistant｜><think>\n",
        "<｜begin▁of▁sentence｜><｜User｜>Solve 3x = 15.<｜Assistant｜><think>\n",
    ]
    llm.generate(warmup_prompts, warmup_params, use_tqdm=False)
    torch.cuda.synchronize()
    print("Warmup complete.", flush=True)

    task_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.0,
        max_tokens=32768,
        seed=20260816,
    )

    # Run Condition A (Single Stream)
    print(f"\n--- Condition A: Balanced Interactive Single-Stream (Concurrency=1, R={args.min_repetitions}) ---", flush=True)
    cond_a_results = []
    for rep in range(1, args.min_repetitions + 1):
        res = run_condition_a_single_stream(
            llm, prompts_a, task_params, rep, args.model_name, args.format, args.raw_dir, gpu_meta, weights_memory_gb
        )
        cond_a_results.append(res)

    # Check Condition A CV
    a_toks = [r["output_tokens_per_second"] for r in cond_a_results]
    mean_a = float(np.mean(a_toks))
    std_a = float(np.std(a_toks))
    cv_a = (std_a / mean_a * 100.0) if mean_a > 0 else 0.0
    print(f"Condition A CV: {cv_a:.2f}% (Threshold: 3.0%)", flush=True)
    if cv_a > 3.0:
        print("CV > 3.0%: Executing 2 additional Condition A replicates (Reps 4 & 5)...", flush=True)
        for rep in range(args.min_repetitions + 1, args.min_repetitions + 3):
            res = run_condition_a_single_stream(
                llm, prompts_a, task_params, rep, args.model_name, args.format, args.raw_dir, gpu_meta, weights_memory_gb
            )
            cond_a_results.append(res)

    # Run Condition B (Batched Throughput with max_num_seqs=8)
    print(f"\n--- Condition B: Pinned Concurrency Batched Throughput (max_num_seqs=8, R={args.min_repetitions}) ---", flush=True)
    cond_b_results = []
    for rep in range(1, args.min_repetitions + 1):
        res = run_condition_b_batched_throughput(
            llm, prompts_b, task_params, rep, args.model_name, args.format, args.raw_dir, gpu_meta, weights_memory_gb
        )
        cond_b_results.append(res)

    # Check Condition B CV
    b_toks = [r["output_tokens_per_second"] for r in cond_b_results]
    mean_b = float(np.mean(b_toks))
    std_b = float(np.std(b_toks))
    cv_b = (std_b / mean_b * 100.0) if mean_b > 0 else 0.0
    print(f"Condition B CV: {cv_b:.2f}% (Threshold: 3.0%)", flush=True)
    if cv_b > 3.0:
        print("CV > 3.0%: Executing 2 additional Condition B replicates (Reps 4 & 5)...", flush=True)
        for rep in range(args.min_repetitions + 1, args.min_repetitions + 3):
            res = run_condition_b_batched_throughput(
                llm, prompts_b, task_params, rep, args.model_name, args.format, args.raw_dir, gpu_meta, weights_memory_gb
            )
            cond_b_results.append(res)

    # Run Microbenchmark
    print("\n--- Secondary Microbenchmark: Fixed-Token (512 tokens) ---", flush=True)
    run_fixed_token_microbenchmark(
        llm, prompts_b, args.model_name, args.format, args.raw_dir, gpu_meta, weights_memory_gb
    )

    print("\nConfirmation benchmark completed successfully!", flush=True)


if __name__ == "__main__":
    main()

"""Publication mode batch-size guard."""

from pathlib import Path

import pytest

from src.runners.inference_session import ConfigurationError, assert_publication_batch_size

ROOT = Path(__file__).resolve().parents[1]


def test_publication_mode_rejects_batch_gt_one():
    with pytest.raises(ConfigurationError):
        assert_publication_batch_size(4, publication=True)


def test_publication_env_rejects_batch_gt_one(monkeypatch):
    monkeypatch.setenv("QREASON_PUBLICATION_MODE", "1")
    with pytest.raises(ConfigurationError):
        assert_publication_batch_size(2, publication=False)


def test_hpc_launcher_exports_publication_mode():
    launcher = ROOT / "scripts/hpc/run_hpc_2a100_publication.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "QREASON_PUBLICATION_MODE=1" in text
    assert "VLLM_BATCH_INVARIANT=1" in text
    assert "--publication" in text


def test_hpc_launcher_preserves_slurm_gpu_allocation():
    launcher = ROOT / "scripts/hpc/run_hpc_2a100_publication.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "cuda_visible_for_gpu()" in text
    assert 'cuda_visible_for_gpu "$gpu_id"' in text
    assert 'CUDA_VISIBLE_DEVICES="$cuda_devices"' in text
    assert 'CUDA_VISIBLE_DEVICES="$gpu_id"' not in text


def test_hpc_submitter_exports_resolved_repo_path():
    submitter = ROOT / "scripts/hpc/submit_hpc_blocks.sh"
    text = submitter.read_text(encoding="utf-8")
    assert "export QR" in text
    assert "QR=${QR}" in text


def test_hpc_submitter_defaults_to_split_2gpu_jobs():
    submitter = ROOT / "scripts/hpc/submit_hpc_blocks.sh"
    text = submitter.read_text(encoding="utf-8")
    assert "QREASON_SUBMIT_2GPU_MODE:-split" in text
    assert "submit_split_2gpu" in text
    assert "exclusive_block|block" in text
    assert "submit_2gpu_block" in text


def test_hpc_submitter_supports_node_excludes():
    submitter = ROOT / "scripts/hpc/submit_hpc_blocks.sh"
    text = submitter.read_text(encoding="utf-8")
    assert "QREASON_SLURM_EXCLUDE" in text
    assert 'SBATCH_EXCLUDE_ARGS=(--exclude="${QREASON_SLURM_EXCLUDE}")' in text
    assert '"${SBATCH_EXCLUDE_ARGS[@]}"' in text


def test_hpc_launcher_checks_free_gpu_memory_before_vllm():
    launcher = ROOT / "scripts/hpc/run_hpc_2a100_publication.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "QREASON_MIN_FREE_GPU_MB:-70000" in text
    assert "check_gpu_free_memory" in text
    assert "nvidia-smi --id" in text
    assert 'check_gpu_free_memory "$gpu_id" "$cuda_devices"' in text


def test_hpc_launcher_requeues_busy_gpu_preflight():
    launcher = ROOT / "scripts/hpc/run_hpc_2a100_publication.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "QREASON_GPU_PREFLIGHT_REQUEUE:-1" in text
    assert "QREASON_GPU_PREFLIGHT_REQUEUE_MAX:-240" in text
    assert 'scontrol requeue "$SLURM_JOB_ID"' in text
    assert "exit 0" in text


def test_hpc_archive_guard_uses_active_python():
    guard = ROOT / "scripts/hpc/09_assert_fresh_archive.sh"
    text = guard.read_text(encoding="utf-8")
    assert 'python scripts/hpc/09_assert_fresh_archive.py --archive "$ROOT"' in text
    assert "python3 scripts/hpc/09_assert_fresh_archive.py" not in text


def test_normal_mode_allows_batch():
    assert_publication_batch_size(4, publication=False)

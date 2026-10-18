from __future__ import annotations

import numpy as np

from scripts.eval.eval_onnx_pipeline_golden_parity import _aggregate_component, compare
from scripts.eval.eval_onnx_composed_pipeline_parity import _compare_component


def test_golden_parity_compare_accepts_close_outputs():
    reference = np.array([[1.0, -2.0, 0.5]], dtype=np.float32)
    actual = reference + np.array([[1.0e-5, -2.0e-5, 1.0e-5]], dtype=np.float32)
    result = compare(reference, actual)
    assert result["passed"] is True
    assert result["shape_match"] is True


def test_golden_parity_compare_rejects_shape_mismatch():
    result = compare(np.zeros((1, 3), dtype=np.float32), np.zeros((1, 2), dtype=np.float32))
    assert result["passed"] is False
    assert result["shape_match"] is False


def test_golden_parity_aggregate_requires_every_sample_to_pass():
    good = compare(np.ones((1, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float32))
    bad = compare(np.zeros((1, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float32))
    result = _aggregate_component([good, bad])
    assert result["sample_count"] == 2
    assert result["passed"] is False
    assert result["relative_l2_error"] > 0.0


def test_composed_parity_keeps_strict_result_and_records_bounded_tolerance():
    reference = np.asarray([[1.0, 2.0]], dtype=np.float32)
    actual = np.asarray([[1.008, 2.004]], dtype=np.float32)

    result = _compare_component(reference, actual)

    assert result["strict_passed"] is False
    assert result["passed"] is True
    assert result["deployment_tolerance"]["relative_l2_error"] == 2.0e-3

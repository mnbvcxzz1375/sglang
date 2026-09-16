"""KV checksum with TP > 1, where the mismatch all-reduce actually executes.

The TP1 coverage in `test_disaggregation_kv_checksum.py` cannot exercise
`_all_reduce_kv_checksum_mismatches` at all: with one rank per engine the
collective takes its `world_size == 1` early return every time. Corruption is
per-rank by nature, so the reduce is what keeps a single rank's mismatch from
splitting the waiting queue and hanging the next collective -- it needs a
multi-rank run to mean anything.

Prefill on GPUs 0-1, decode on GPUs 2-3.
"""

import unittest
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

# base-c is the stage that owns the 4-gpu-h100 suite; base-b has no such
# runner, and register_cuda_ci is AST-parsed, so a bad pair fails silently.
register_cuda_ci(est_time=600, stage="base-c", runner_config="4-gpu-h100")

_CHECKSUM_ARGS = ["--disaggregation-enable-kv-checksum"]


def _decode_body(response: requests.Response):
    try:
        return response.json()
    except ValueError:
        return response.text


def _is_abort_result(status_code: int, body) -> bool:
    """An aborted request can surface either way.

    Same contract as `test_disaggregation_chunked_prefill_abort._is_abort_result`:
    a 200 carrying `meta_info.finish_reason.type == "abort"`, or a 5xx whose
    body names the abort. Asserting on the status code alone passes or fails
    for the wrong reason.
    """
    if status_code == 200:
        reason = (
            body.get("meta_info", {}).get("finish_reason", {})
            if isinstance(body, dict)
            else {}
        )
        return isinstance(reason, dict) and reason.get("type") == "abort"
    if status_code not in (500, 503):
        return False
    text = body if isinstance(body, str) else str(body)
    return "abort" in text.lower()


class _TP2Base(PDDisaggregationServerBase):
    prefill_tp_size = 2
    decode_tp_size = 2
    decode_base_gpu_id = 2


class TestDisaggregationKVChecksumTP2(_TP2Base):
    """Healthy traffic under TP2: the reduce runs every batch and drops nothing."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = list(_CHECKSUM_ARGS)
        cls.extra_decode_args = list(_CHECKSUM_ARGS)
        cls.launch_all()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.lb_url,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")
        self.assertGreater(metrics["score"], 0.62)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


class TestDisaggregationKVChecksumTP2SingleRankCorruption(_TP2Base):
    """One rank's fault must abort the request on every rank, not hang.

    The injection is probabilistic per request and evaluated independently on
    each decode rank, so a mismatch seen by one rank and not its peer is the
    normal case here -- which is exactly the split the all-reduce exists to
    prevent. What must not happen is a hang: the engines stay responsive and
    later requests still succeed.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = list(_CHECKSUM_ARGS)
        cls.extra_decode_args = list(_CHECKSUM_ARGS)
        # Low enough that ranks routinely disagree about a given request.
        cls.extra_decode_env = {"SGLANG_TEST_DISAGG_KV_CORRUPT_PROB": "0.5"}
        cls.launch_all()

    def test_single_rank_mismatch_does_not_hang(self):
        outcomes = []
        for _ in range(8):
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                },
                timeout=120,
            )
            outcomes.append(
                _is_abort_result(response.status_code, _decode_body(response))
            )
        # Some requests are aborted, which is the point; the engines survive and
        # keep serving, which is what the reduce buys. With p=0.5 evaluated
        # independently per decode rank, P(no abort in 8) is about 1.5e-5.
        self.assertTrue(any(outcomes), outcomes)
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


if __name__ == "__main__":
    unittest.main()

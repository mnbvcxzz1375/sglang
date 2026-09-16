"""End-to-end coverage for --disaggregation-enable-kv-checksum.

The unit tests drive fake pools, so they cannot tell whether the two engines
actually agree on what bytes a handoff covers. Only a real transfer can. Two
things have to hold for the flag to be worth turning on: accuracy is unchanged
and no healthy request is aborted, and the digest still lines up when the
handoff is not the simple whole-prompt case.

The decode-side radix cache is on deliberately: a prefix hit shrinks the
handoff to a suffix, so the digest has to cover exactly the transferred range
on both sides rather than the whole prompt.
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

register_cuda_ci(est_time=600, stage="base-b", runner_config="2-gpu-large")

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


class TestDisaggregationKVChecksum(PDDisaggregationServerBase):
    """Checksum on, healthy transfers: nothing may change but the cost."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        # Chunked prefill, so the digest spans a handoff sent as several chunks
        # and is taken on the last one.
        cls.extra_prefill_args = list(_CHECKSUM_ARGS) + [
            "--chunked-prefill-size",
            "1024",
        ]
        # Decode-side radix cache, so a prefix hit makes the prefill send only a
        # suffix; the digest must then cover [decode_prefix_len, end) on both
        # sides, not [0, end).
        cls.extra_decode_args = list(_CHECKSUM_ARGS) + [
            "--disaggregation-decode-enable-radix-cache",
        ]
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

    def test_no_healthy_request_is_aborted(self):
        for prompt in (
            "The capital of France is",
            "Write a haiku about disaggregated inference:",
            "Count from one to twenty: " * 40,
        ):
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)
            body = _decode_body(response)
            self.assertFalse(
                _is_abort_result(response.status_code, body),
                f"healthy transfer was aborted: {response.text}",
            )
            self.assertTrue(body.get("text"), response.text)
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)

    def test_partial_transfer_after_decode_prefix_hit(self):
        """A decode prefix hit shrinks the handoff to a suffix of the prompt.

        The repeats transfer far fewer tokens than the first request, so if the
        digest covered the whole prompt the cached prefix -- bytes from an
        earlier prefill run, not this one -- would not line up.
        """
        prompt = "Summarize the following. " + "The quick brown fox jumps. " * 300
        outputs = []
        for _ in range(3):
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 24},
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)
            body = _decode_body(response)
            self.assertFalse(
                _is_abort_result(response.status_code, body),
                f"partial transfer was aborted: {response.text}",
            )
            outputs.append(body["text"])
        # Same prompt, greedy: a partial transfer must reconstruct the same KV.
        self.assertEqual(len(set(outputs)), 1, outputs)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


class TestDisaggregationKVChecksumOneSided(PDDisaggregationServerBase):
    """Enabled on decode only: skipped with a log, never an abort or a crash.

    The decode registers one more aux buffer than the prefill, and the aux list
    is matched positionally. This used to run off the end of the prefill's
    list; now both sides transfer what they share and the decode reads "no
    digest" and skips.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_decode_args = list(_CHECKSUM_ARGS)
        cls.launch_all()

    def test_serves_normally_with_one_sided_enable(self):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(
            _is_abort_result(response.status_code, _decode_body(response)),
            f"one-sided enable must skip the check, not abort: {response.text}",
        )
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


class TestDisaggregationKVChecksumDetectsCorruption(PDDisaggregationServerBase):
    """The true positive: a corrupted handoff must be caught, not decoded.

    Without this the other cases all pass if `compute()` returned a constant.
    The injected fault clobbers a landed KV row the way a slot reused
    mid-write would, on the decode side, after the transfer completed.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = list(_CHECKSUM_ARGS)
        cls.extra_decode_args = list(_CHECKSUM_ARGS)
        cls.extra_decode_env = {"SGLANG_TEST_DISAGG_KV_CORRUPT_PROB": "1.0"}
        cls.launch_all()

    def test_corrupted_kv_is_rejected(self):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=120,
        )
        # The request must be aborted rather than decode against the wrong KV.
        # The engines stay up: a checksum mismatch drops one request, not the
        # server.
        self.assertTrue(
            _is_abort_result(response.status_code, _decode_body(response)),
            f"expected an abort, got {response.status_code}: {response.text}",
        )
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


if __name__ == "__main__":
    unittest.main()

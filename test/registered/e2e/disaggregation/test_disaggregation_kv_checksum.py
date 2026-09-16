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
            self.assertNotIn("checksum", response.text.lower())
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
            outputs.append(response.json()["text"])
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
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


if __name__ == "__main__":
    unittest.main()

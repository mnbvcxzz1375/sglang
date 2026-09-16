from __future__ import annotations

import hashlib
import logging
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from sglang.kernels.ops.memory.adler32 import adler32_strided_checksum
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

NestedInts = Sequence[Sequence[int]]

# State components whose rows are addressed by whole-sequence KV pages, in the
# pool's own id space. Mirrors the `_full_kv_pages_payload` entries of the
# transfer's payload table.
_FULL_KV_PAGE_STATES = (
    StateType.DSA,
    StateType.QSA_COMPRESSED,
    StateType.MINIMAX_INDEX_K,
    StateType.BLOCK_SCALE,
)

# State components addressed by sliding-window pages.
_SWA_PAGE_STATES = (StateType.SWA, StateType.BLOCK_SCALE_SWA)

# StateType.DSA_TAIL is deliberately absent. `get_dsa_tail_state_indices`
# returns a *descriptor* -- [req_pool_idx, start_phys, first_n, 0, second_n,
# tail_size] -- that the transfer backends decode into byte blocks, not a list
# of rows. Feeding it to the digest would read this request's row plus five
# rows belonging to other, concurrently mutating requests, and `req_pool_idx`
# differs between the two engines anyway. Until it has a real index mapping it
# is excluded on both sides, which is what the builder table's fallthrough
# does.

_warned_unaddressable: set = set()

# Bump when the digest's meaning changes, so peers running different builds
# compare signatures rather than digests and skip instead of aborting.
_SIGNATURE_VERSION = 1


def is_health_check_req(req: Req) -> bool:
    rid = req.rid
    return isinstance(rid, str) and rid.startswith(HEALTH_CHECK_RID_PREFIX)


def _to_page_indices_gpu(idx: torch.Tensor, page_size: int) -> torch.Tensor:
    idx = idx.to(torch.int64).contiguous().reshape(-1)
    if page_size == 1:
        return idx
    return (idx[::page_size] // page_size).contiguous()


def _as_index_tensor(values, device: torch.device) -> torch.Tensor:
    """Normalize a transfer payload (list / ndarray / tensor) to GPU int64."""
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.int64).contiguous().reshape(-1)
    if isinstance(values, np.ndarray):
        flat = values.reshape(-1)
    else:
        flat = np.asarray(list(values), dtype=np.int64).reshape(-1)
    return torch.as_tensor(flat, dtype=torch.int64, device=device).contiguous()


def kv_page_indices_for_request(
    scheduler, req: Req, start_idx: int, end_idx: int
) -> torch.Tensor:
    """Main-KV page ids over exactly the range the transfer sends.

    Two things have to match the send path in `_send_kv_chunk` or the two
    sides digest different bytes: the range starts at `start_idx` (the prefill
    skips `[0, decode_prefix_len)`, which decode filled from its own cache),
    and the ids go through `translate_kv_indices_for_transfer`, because
    `req_to_token` holds VIRTUAL ids on a unified-memory pool while the
    registered buffers are addressed by physical ones.
    """
    page_size = scheduler.token_to_kv_pool_allocator.page_size
    kv_indices = scheduler.req_to_token_pool.req_to_token[
        req.kv.req_pool_idx, start_idx:end_idx
    ]
    kv_indices = scheduler.token_to_kv_pool_allocator.translate_kv_indices_for_transfer(
        kv_indices
    )
    return _to_page_indices_gpu(kv_indices, page_size)


def _state_index_builders(
    scheduler, req: Req, seq_len: int, start_idx: int, device: torch.device
) -> Dict[StateType, Callable[[], Optional[torch.Tensor]]]:
    """Per-``StateType`` index builders, mirroring the transfer's payload table.

    Each component is addressed in its own id space, so there is no single
    tensor that indexes all of them. Anything not listed here is left out of
    the digest rather than read at the wrong offsets -- see
    `state_indices_for_request`.
    """
    allocator = scheduler.token_to_kv_pool_allocator
    pool = allocator.get_kvcache()
    page_size = allocator.page_size
    req_pool_idx = int(req.kv.req_pool_idx)

    def _mamba() -> Optional[torch.Tensor]:
        mapping = getattr(
            scheduler.req_to_token_pool, "req_index_to_mamba_index_mapping", None
        )
        if mapping is None:
            return None
        rows = mapping[req_pool_idx]
        translate = getattr(
            scheduler.req_to_token_pool, "translate_mamba_indices", None
        )
        if translate is not None:
            rows = translate(rows)
        return _as_index_tensor(rows, device)

    def _swa_pages() -> Optional[torch.Tensor]:
        window_size = getattr(scheduler, "sliding_window_size", None)
        if window_size is None:
            return None
        # Same base as the send path: the prefix decode already holds is not
        # transferred, so it must not be digested either.
        window_start = max(start_idx, seq_len - window_size)
        window_start = (window_start // page_size) * page_size
        window_full = scheduler.req_to_token_pool.req_to_token[
            req_pool_idx, window_start:seq_len
        ]
        # translate_swa_indices_for_transfer, not translate_loc_from_full_to_swa:
        # the latter returns kernel-facing ids, the buffers are addressed by
        # physical ones.
        window_swa = allocator.translate_swa_indices_for_transfer(window_full)
        return _to_page_indices_gpu(window_swa, page_size)

    def _full_kv_pages() -> Optional[torch.Tensor]:
        # Whole sequence and untranslated, matching `_full_kv_pages_payload`:
        # these pools are not virtual-id remapped.
        device_page_size = getattr(pool, "page_size", page_size)
        kv_full = scheduler.req_to_token_pool.req_to_token[req_pool_idx, :seq_len]
        return _to_page_indices_gpu(kv_full, device_page_size)

    def _qsa_pending() -> Optional[torch.Tensor]:
        from sglang.srt.disaggregation.utils import get_qsa_pending_state_indices

        return _as_index_tensor(get_qsa_pending_state_indices(req), device)

    def _swa_ring() -> Optional[torch.Tensor]:
        ring_stride = getattr(pool, "unified_swa_ring_size", None)
        window_size = getattr(pool, "unified_swa_window", None)
        if ring_stride is None or window_size is None:
            return None
        window_start = max(0, seq_len - window_size)
        positions = np.arange(window_start, seq_len, dtype=np.int64)
        ring_rows = req_pool_idx * ring_stride + (positions % ring_stride)
        return _as_index_tensor(ring_rows, device)

    def _c128_state() -> Optional[torch.Tensor]:
        from sglang.srt.disaggregation.utils import (
            get_dsv4_c128_state_indices,
            is_dsv4_c128_online_enabled,
        )

        online = is_dsv4_c128_online_enabled()
        get_ring_size = getattr(pool, "get_ring_size", None)
        if not online and get_ring_size is None:
            return None
        ring_size = 1 if online else get_ring_size(128)
        # Request-scoped: keyed by the logical input length, as the transfer is.
        return _as_index_tensor(
            get_dsv4_c128_state_indices(
                req_pool_idx,
                len(req.origin_input_ids),
                online=online,
                ring_size=ring_size,
            ),
            device,
        )

    builders: Dict[StateType, Callable[[], Optional[torch.Tensor]]] = {
        StateType.MAMBA: _mamba,
        StateType.QSA_PENDING: _qsa_pending,
        StateType.SWA_RING: _swa_ring,
        StateType.DSV4_REQUEST_STATE: _c128_state,
    }
    for st in _FULL_KV_PAGE_STATES:
        builders[st] = _full_kv_pages
    for st in _SWA_PAGE_STATES:
        builders[st] = _swa_pages
    return builders


def state_indices_for_request(
    scheduler,
    req: Req,
    seq_len: int,
    state_types: Sequence[StateType],
    start_idx: int = 0,
    device: Optional[torch.device] = None,
) -> List[Optional[torch.Tensor]]:
    """One index tensor per state component, parallel to ``state_types``.

    Dispatch is on the component's own ``StateType``, not on the pool class:
    a pool contributes several components with unrelated id spaces (a
    block-scaled SWA pool ships SWA, BLOCK_SCALE and BLOCK_SCALE_SWA), and the
    pool class says nothing about how any one of them is addressed.
    """
    if not state_types:
        return []
    if device is None:
        device = scheduler.req_to_token_pool.req_to_token.device
    builders = _state_index_builders(scheduler, req, seq_len, start_idx, device)
    out: List[Optional[torch.Tensor]] = []
    for st in state_types:
        builder = builders.get(st)
        if builder is None:
            if st not in _warned_unaddressable:
                _warned_unaddressable.add(st)
                logger.warning(
                    "KV checksum: state component %s has no index mapping; it is "
                    "excluded from the digest on both sides.",
                    st,
                )
            out.append(None)
            continue
        try:
            out.append(builder())
        except Exception as exc:  # pragma: no cover - defensive
            if st not in _warned_unaddressable:
                _warned_unaddressable.add(st)
                logger.warning(
                    "KV checksum: could not index state component %s (%s); it is "
                    "excluded from the digest on both sides.",
                    st,
                    exc,
                )
            out.append(None)
    return out


class KvChecksumComputer:
    """Adler-32 over the KV (and addressable state) a handoff covers.

    State components stay *nested*: ``state_data_ptrs[i]`` is the buffer list
    of ``state_types[i]``, and each gets its own index tensor. Flattening them
    into one list forces one index tensor across every component, which reads
    the wrong rows for any pool with more than one.
    """

    def __init__(
        self,
        device: torch.device,
        kv_data_ptrs: Sequence[int],
        kv_item_lens: Sequence[int],
        state_types: Sequence[StateType] = (),
        state_data_ptrs: NestedInts = (),
        state_item_lens: NestedInts = (),
        page_size: int = 1,
    ):
        assert len(kv_data_ptrs) == len(kv_item_lens)
        assert len(kv_data_ptrs) > 0
        self._device = torch.device(device)
        self._kv_data_ptrs = [int(ptr) for ptr in kv_data_ptrs]
        self._kv_item_lens = [int(item_len) for item_len in kv_item_lens]
        assert len(state_data_ptrs) == len(state_item_lens)
        assert len(state_types) == len(state_data_ptrs)
        self.state_types = list(state_types)
        self._state_data_ptrs = [[int(ptr) for ptr in ptrs] for ptrs in state_data_ptrs]
        self._state_item_lens = [
            [int(item_len) for item_len in lens] for lens in state_item_lens
        ]
        for ptrs, lens in zip(self._state_data_ptrs, self._state_item_lens):
            assert len(ptrs) == len(lens)
        self.signature = self._compute_signature(page_size)

    def _compute_signature(self, page_size: int) -> int:
        """Identify the KV layout this digest is taken over.

        Two sides can only compare digests if they cover the same bytes. A
        prefill with a different TP width has different per-page item lengths;
        a layer-sharded or pipeline-parallel prefill owns a subset of the
        buffers; a peer with the feature off sends nothing. Shipping this
        alongside the digest turns all of those into a logged skip instead of
        a mismatch on every request.
        """
        parts = [
            str(_SIGNATURE_VERSION),
            str(page_size),
            str(len(self._kv_data_ptrs)),
            ",".join(str(x) for x in self._kv_item_lens),
            ",".join(str(getattr(st, "value", st)) for st in self.state_types),
            ";".join(",".join(str(x) for x in lens) for lens in self._state_item_lens),
        ]
        digest = hashlib.blake2b("|".join(parts).encode(), digest_size=8).digest()
        # Keep it positive in the int64 metadata slot, and never 0 -- 0 is the
        # "no digest was written" sentinel.
        return (int.from_bytes(digest, "big") & ((1 << 62) - 1)) | 1

    def compute(
        self,
        kv_page_indices_gpu: torch.Tensor,
        state_indices_gpu: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> int:
        assert kv_page_indices_gpu.is_cuda and kv_page_indices_gpu.is_contiguous()
        all_ptrs = list(self._kv_data_ptrs)
        all_lens = list(self._kv_item_lens)
        all_indices: List[torch.Tensor] = [kv_page_indices_gpu] * len(
            self._kv_data_ptrs
        )
        if self._state_data_ptrs:
            indices = list(state_indices_gpu or [])
            assert len(indices) == len(self._state_data_ptrs), (
                "state index list must be parallel to state_types"
            )
            for ptrs, lens, idx in zip(
                self._state_data_ptrs, self._state_item_lens, indices
            ):
                if idx is None or idx.numel() == 0:
                    # Component this side cannot address; excluded on both sides.
                    continue
                assert idx.is_cuda and idx.is_contiguous()
                all_ptrs += ptrs
                all_lens += lens
                all_indices += [idx] * len(ptrs)
        return adler32_strided_checksum(all_ptrs, all_lens, all_indices)


def corrupt_one_kv_row_for_test(scheduler, kv_page_indices_gpu: torch.Tensor) -> bool:
    """Clobber one landed KV row, the way a slot reused mid-write would.

    Test-only, behind SGLANG_TEST_DISAGG_KV_CORRUPT_PROB. Writes through the
    pool's own buffers so the digest sees exactly what a real fault would
    leave behind. Returns whether anything was corrupted.
    """
    if kv_page_indices_gpu.numel() == 0:
        return False
    pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    for name in ("k_buffer", "kv_buffer", "v_buffer"):
        buffers = getattr(pool, name, None)
        if not buffers:
            continue
        row = int(kv_page_indices_gpu[0].item())
        if row >= buffers[0].shape[0]:
            continue
        buffers[0][row] += 1
        return True
    return False

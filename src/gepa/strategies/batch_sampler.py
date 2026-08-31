# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random
from collections import Counter
from collections.abc import Mapping
from typing import Any, Protocol

from gepa.core.adapter import DataInst
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState


class BatchSampler(Protocol[DataId, DataInst]):
    """Yields the minibatch of trainset ids to propose from.

    Multi-proposal sampling strategies call ``next_minibatch_ids`` once per
    task within a single iteration (``state.i`` unchanged between calls).
    Implementations should return a *different* minibatch on each repeated
    call within an iteration, so parallel proposal tasks don't all share one
    minibatch.
    """

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]: ...


class EpochShuffledBatchSampler(BatchSampler[DataId, DataInst]):
    """
    Mirrors the original batching logic:
    - Shuffle ids each epoch
    - Pad to minibatch size with least frequent ids
    - Deterministic via state.rng1
    """

    def __init__(self, minibatch_size: int, rng: random.Random | None = None):
        self.minibatch_size = minibatch_size
        self.shuffled_ids: list[DataId] = []
        self.epoch = -1
        self.id_freqs = Counter()
        self.last_trainset_size = 0
        self._current_iteration: int | None = None
        self._calls_in_iteration = 0
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def get_state(self) -> dict[str, Any]:
        """Return the current epoch permutation and sampling cursor.

        The RNG is intentionally excluded because the engine checkpoints the
        single shared run RNG separately.

        Returns:
            Serializable sampler state needed to continue an epoch exactly.
        """
        state = {
            "minibatch_size": self.minibatch_size,
            "shuffled_ids": list(self.shuffled_ids),
            "epoch": self.epoch,
            "id_freqs": dict(self.id_freqs),
            "last_trainset_size": self.last_trainset_size,
            "current_iteration": self._current_iteration,
            "calls_in_iteration": self._calls_in_iteration,
        }
        return state

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore an epoch permutation and cursor from a durable checkpoint.

        Args:
            state: Snapshot previously returned by :meth:`get_state`.

        Raises:
            TypeError: A persisted collection has an incompatible type.
            ValueError: The snapshot was created with a different minibatch
                size.
        """
        if int(state.get("minibatch_size", -1)) != self.minibatch_size:
            raise ValueError("Persisted batch sampler minibatch_size does not match the current run")
        shuffled_ids = state.get("shuffled_ids", [])
        id_freqs = state.get("id_freqs", {})
        if not isinstance(shuffled_ids, list):
            raise TypeError("Persisted shuffled_ids must be a list")
        if not isinstance(id_freqs, Mapping):
            raise TypeError("Persisted id_freqs must be a mapping")
        self.shuffled_ids = list(shuffled_ids)
        self.epoch = int(state.get("epoch", -1))
        self.id_freqs = Counter(id_freqs)
        self.last_trainset_size = int(state.get("last_trainset_size", 0))
        current_iteration = state.get("current_iteration")
        self._current_iteration = None if current_iteration is None else int(current_iteration)
        self._calls_in_iteration = int(state.get("calls_in_iteration", 0))

    def _update_shuffled(self, loader: DataLoader[DataId, DataInst]):
        all_ids = list(loader.all_ids())
        trainset_size = len(loader)
        self.last_trainset_size = trainset_size

        if trainset_size == 0:
            self.shuffled_ids = []
            self.id_freqs = Counter()
            return

        self.shuffled_ids = list(all_ids)
        self.rng.shuffle(self.shuffled_ids)
        self.id_freqs = Counter(self.shuffled_ids)

        mod = trainset_size % self.minibatch_size
        num_to_pad = (self.minibatch_size - mod) if mod != 0 else 0
        if num_to_pad > 0:
            for _ in range(num_to_pad):
                selected_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_ids.append(selected_id)
                self.id_freqs[selected_id] += 1

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]:
        trainset_size = len(loader)
        if trainset_size == 0:
            raise ValueError("Cannot sample a minibatch from an empty loader.")

        # Repeated calls within one iteration (multi-proposal sampling
        # strategies request one minibatch per task) advance one chunk each,
        # so tasks in the same iteration get distinct minibatches. The first
        # call of an iteration is unchanged from the classic behavior. Chunks
        # wrap around, so distinctness holds only while the iteration's call
        # count stays within len(shuffled_ids) / minibatch_size.
        if state.i == self._current_iteration:
            self._calls_in_iteration += 1
        else:
            self._current_iteration = state.i
            self._calls_in_iteration = 0

        base_idx = state.i * self.minibatch_size
        curr_epoch = 0 if self.epoch == -1 else base_idx // max(len(self.shuffled_ids), 1)

        needs_refresh = not self.shuffled_ids or trainset_size != self.last_trainset_size or curr_epoch > self.epoch
        if needs_refresh:
            self.epoch = curr_epoch
            self._update_shuffled(loader)

        assert len(self.shuffled_ids) >= self.minibatch_size
        assert len(self.shuffled_ids) % self.minibatch_size == 0

        # The epoch bookkeeping above uses the un-offset base_idx (constant
        # within an iteration), so repeat calls never trigger a reshuffle and
        # the shuffle sequence stays identical to the single-call path.
        base_idx = (base_idx + self._calls_in_iteration * self.minibatch_size) % len(self.shuffled_ids)
        end_idx = base_idx + self.minibatch_size
        assert end_idx <= len(self.shuffled_ids)
        return self.shuffled_ids[base_idx:end_idx]

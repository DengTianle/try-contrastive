from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, Protocol

from torch.utils.data import Sampler


class GroupedDataset(Protocol):
    groups: list[dict[str, Any]]

    def __len__(self) -> int:
        ...


class DifferentSongBatchSampler(Sampler[list[int]]):
    """Plan balanced batches containing at most one segment from each song."""

    def __init__(
        self,
        dataset: GroupedDataset,
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.indices_by_song: dict[str, list[int]] = defaultdict(list)
        for index, group in enumerate(self.dataset.groups):
            self.indices_by_song[group["anchor"]["dali_id"]].append(index)

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible batch plan for an epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def _full_batch_count(self) -> int:
        """Return the maximum feasible number of full song-distinct batches."""
        upper_bound = len(self.dataset) // self.batch_size
        song_sizes = [len(indices) for indices in self.indices_by_song.values()]
        for batch_count in range(upper_bound, 0, -1):
            usable_examples = sum(min(size, batch_count) for size in song_sizes)
            if usable_examples >= batch_count * self.batch_size:
                return batch_count
        return 0

    def _batch_count(self) -> int:
        if not self.indices_by_song:
            return 0
        if self.drop_last:
            return self._full_batch_count()
        return max(
            math.ceil(len(self.dataset) / self.batch_size),
            max(len(indices) for indices in self.indices_by_song.values()),
        )

    def _selected_indices_by_song(
        self,
        batch_count: int,
        rng: random.Random,
    ) -> dict[str, list[int]]:
        shuffled_by_song: dict[str, list[int]] = {}
        for song, song_indices in self.indices_by_song.items():
            indices = list(song_indices)
            rng.shuffle(indices)
            shuffled_by_song[song] = indices

        if not self.drop_last:
            return shuffled_by_song

        # A full batch can use a song at most once, so no song can contribute
        # more than batch_count examples. Randomly choose among all usable
        # examples when more than the required full-batch capacity is available.
        usable: list[tuple[str, int]] = []
        for song, indices in shuffled_by_song.items():
            usable.extend((song, index) for index in indices[:batch_count])
        rng.shuffle(usable)
        usable = usable[: batch_count * self.batch_size]

        selected: dict[str, list[int]] = defaultdict(list)
        for song, index in usable:
            selected[song].append(index)
        return dict(selected)

    def _plan_batches(self) -> list[list[int]]:
        batch_count = self._batch_count()
        if batch_count == 0:
            return []

        rng = random.Random(f"different-song-batches:{self.seed}:{self.epoch}")
        indices_by_song = self._selected_indices_by_song(batch_count, rng)
        batches: list[list[int]] = [[] for _ in range(batch_count)]

        # Process larger songs first. Randomizing before the stable sort and when
        # ordering equally loaded batches changes pairings between epochs without
        # changing the number or size profile of the batches.
        songs = list(indices_by_song)
        rng.shuffle(songs)
        songs.sort(key=lambda song: len(indices_by_song[song]), reverse=True)
        for song in songs:
            available_batches = list(range(batch_count))
            rng.shuffle(available_batches)
            available_batches.sort(key=lambda index: len(batches[index]))
            for index, batch_index in zip(indices_by_song[song], available_batches):
                batches[batch_index].append(index)

        for batch in batches:
            rng.shuffle(batch)
        rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._plan_batches()

    def __len__(self) -> int:
        return self._batch_count()

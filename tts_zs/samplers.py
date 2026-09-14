import random

from torch.utils.data import BatchSampler


class DistinctModelBatchSampler(BatchSampler):
    def __init__(self, samples: list, batch_size: int, seed: int = 0):
        super().__init__(sampler=None, batch_size=batch_size, drop_last=True)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.by_class: dict = {}
        for i, (c, _) in enumerate(samples):
            self.by_class.setdefault(c, []).append(i)
        self.n_classes = len(self.by_class)
        if batch_size > self.n_classes:
            raise ValueError(
                f"batch_size={batch_size} exceeds #classes={self.n_classes}; "
                "cannot form a same-class-free batch"
            )
        self.total_samples = sum(len(v) for v in self.by_class.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools = {c: rng.sample(idxs, k=len(idxs)) for c, idxs in self.by_class.items()}
        cursors = {c: 0 for c in pools}
        classes = list(pools.keys())
        remaining_total = self.total_samples
        while remaining_total >= self.batch_size:
            available = [c for c in classes if cursors[c] < len(pools[c])]
            if len(available) < self.batch_size:
                break
            chosen = rng.sample(available, k=self.batch_size)
            batch = []
            for c in chosen:
                batch.append(pools[c][cursors[c]])
                cursors[c] += 1
            remaining_total -= self.batch_size
            yield batch

    def __len__(self):
        return self.total_samples // self.batch_size


class DistinctModelBatchSamplerLC(BatchSampler):
    def __init__(
        self,
        samples: list,
        pair_to_model: dict,
        batch_size: int,
        seed: int = 0,
    ):
        super().__init__(sampler=None, batch_size=batch_size, drop_last=True)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        by_model: dict = {}
        for i, (pair_idx, _) in enumerate(samples):
            m = pair_to_model[pair_idx]
            by_model.setdefault(m, []).append(i)
        self.by_model = by_model
        self.n_models = len(by_model)
        if batch_size > self.n_models:
            raise ValueError(
                f"batch_size={batch_size} exceeds #models={self.n_models}; "
                "cannot form a model-free batch"
            )
        self.total_samples = sum(len(v) for v in by_model.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools = {m: rng.sample(idxs, k=len(idxs)) for m, idxs in self.by_model.items()}
        cursors = {m: 0 for m in pools}
        models = list(pools.keys())
        remaining_total = self.total_samples
        while remaining_total >= self.batch_size:
            available = [m for m in models if cursors[m] < len(pools[m])]
            if len(available) < self.batch_size:
                break
            chosen = rng.sample(available, k=self.batch_size)
            batch = []
            for m in chosen:
                batch.append(pools[m][cursors[m]])
                cursors[m] += 1
            remaining_total -= self.batch_size
            yield batch

    def __len__(self):
        return self.total_samples // self.batch_size


class KPerClassBatchSampler(BatchSampler):
    def __init__(self, samples: list, k: int, batch_size: int, seed: int = 0):
        if batch_size % k != 0:
            raise ValueError(f"batch_size={batch_size} not divisible by k={k}")
        super().__init__(sampler=None, batch_size=batch_size, drop_last=True)
        self.k = k
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        by_class: dict = {}
        for i, (c, _) in enumerate(samples):
            by_class.setdefault(c, []).append(i)
        self.by_class = {c: v for c, v in by_class.items() if len(v) >= k}
        n_per_batch = batch_size // k
        if n_per_batch > len(self.by_class):
            raise ValueError(
                f"batch needs {n_per_batch} classes but only {len(self.by_class)} "
                f"have >= {k} clips"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        n_per_batch = self.batch_size // self.k
        pools = {c: rng.sample(v, k=len(v)) for c, v in self.by_class.items()}
        cursors = {c: 0 for c in pools}
        classes = list(pools.keys())
        while True:
            available = [c for c in classes if cursors[c] + self.k <= len(pools[c])]
            if len(available) < n_per_batch:
                break
            chosen = rng.sample(available, k=n_per_batch)
            batch = []
            for c in chosen:
                start = cursors[c]
                batch.extend(pools[c][start : start + self.k])
                cursors[c] += self.k
            yield batch

    def __len__(self):
        n_groups = sum(len(v) // self.k for v in self.by_class.values())
        return n_groups // (self.batch_size // self.k)


class RotatingHoldoutKPerClassSampler(BatchSampler):
    def __init__(
        self,
        samples: list,
        pair_to_model: dict,
        k: int,
        batch_size: int,
        n_holdout: int,
        seed: int = 0,
    ):
        if batch_size % k != 0:
            raise ValueError(f"batch_size={batch_size} not divisible by k={k}")
        super().__init__(sampler=None, batch_size=batch_size, drop_last=True)
        self.samples = samples
        self.pair_to_model = pair_to_model
        self.k = k
        self.batch_size = batch_size
        self.n_holdout = n_holdout
        self.seed = seed
        self.epoch = 0
        self.all_models = sorted({v for v in pair_to_model.values()})
        if n_holdout >= len(self.all_models):
            raise ValueError(
                f"n_holdout={n_holdout} >= n_models={len(self.all_models)}"
            )
        self._rebuild()

    def _rebuild(self):
        rng = random.Random(self.seed + self.epoch)
        virtual_unseen = set(rng.sample(self.all_models, k=self.n_holdout))
        kept = [
            (idx, p)
            for idx, p in self.samples
            if self.pair_to_model[idx] not in virtual_unseen
        ]
        self._inner = KPerClassBatchSampler(
            kept, k=self.k, batch_size=self.batch_size, seed=self.seed + self.epoch
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._rebuild()

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)


class LangStratifiedKPerClassSampler(KPerClassBatchSampler):
    def __init__(
        self,
        samples: list,
        pair_to_lang: dict,
        k: int,
        batch_size: int,
        seed: int = 0,
    ):
        super().__init__(samples, k, batch_size, seed)
        by_lang: dict = {}
        for c in self.by_class:
            lang = pair_to_lang[c]
            by_lang.setdefault(lang, []).append(c)
        self.by_lang = by_lang

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        n_per_batch = self.batch_size // self.k
        pools = {c: rng.sample(v, k=len(v)) for c, v in self.by_class.items()}
        cursors = {c: 0 for c in pools}

        def avail(c):
            return cursors[c] + self.k <= len(pools[c])

        while True:
            chosen = []
            for lang_classes in self.by_lang.values():
                eligible = [c for c in lang_classes if avail(c)]
                take = min(2, len(eligible), n_per_batch - len(chosen))
                if take > 0:
                    chosen.extend(rng.sample(eligible, k=take))
                if len(chosen) >= n_per_batch:
                    break
            if len(chosen) < n_per_batch:
                chosen_set = set(chosen)
                extras = [c for c in pools if avail(c) and c not in chosen_set]
                still_need = n_per_batch - len(chosen)
                if len(extras) < still_need:
                    break
                chosen.extend(rng.sample(extras, k=still_need))
            if len(chosen) < n_per_batch:
                break
            batch = []
            for c in chosen:
                start = cursors[c]
                batch.extend(pools[c][start : start + self.k])
                cursors[c] += self.k
            yield batch

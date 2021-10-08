import torch

from typing import Iterator, Optional, Sized

from torch import Tensor
from torch.utils.data import Sampler


class RandomPairSampler(Sampler[int]):
    r"""Samples element pairs randomly, in sequence.
    The sample order is random element -> it's following one -> another random element -> it's following one -> etc.
    This allows us to have batches of pairs, that appear as a batch of images, instead of batch of tuples for example.
    If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0 or self.num_samples % 2 != 0:
            raise ValueError("num_samples should be an even positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return 2 * len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        # last element has no pair
        n = len(self.data_source) - 1
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        if self.replacement:
            num_rand = (self.num_samples // 2)
            for _ in range(num_rand // 16):
                yield from self._add_pairs(torch.randint(high=n, size=(16,),
                                                         dtype=torch.int64, generator=generator)).tolist()
            yield from self._add_pairs(torch.randint(high=n, size=(num_rand % 16,),
                                                     dtype=torch.int64, generator=generator)).tolist()
        else:
            yield from self._add_pairs(torch.randperm(n, generator=generator)).tolist()

    def __len__(self) -> int:
        return self.num_samples

    @staticmethod
    def _add_pairs(samples: Tensor) -> Tensor:
        pairs = samples + 1
        c = torch.zeros(2 * len(samples), dtype=torch.int64)
        c[::2] = samples  # Index every second row, starting from 0
        c[1::2] = pairs  # Index every second row, starting from 1
        return c

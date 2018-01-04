
import numpy as np
from typing import List


class DynamicTargetDataset(object):
    """An abstract class representing a Dataset where target can be updated on the fly.

    All other NAT datasets should subclass it. All subclasses should override
    ``update_targets``, that provides a method to update the target representation of the dataset.
    """

    def update_targets(self, indexes: List[int], new_targets: np.ndarray):
        raise NotImplementedError

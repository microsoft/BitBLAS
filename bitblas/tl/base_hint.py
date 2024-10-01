# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas.base.roller.hint import Hint
from abc import ABC, abstractmethod
class BaseTLHint(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        raise NotImplementedError(
            f"__repr__ is not implemented"
        )
    
    def from_roller_hint(self, hint: Hint):
        raise NotImplementedError(
            f"from_roller_hint is not implemented"
        )
    
    @abstractmethod
    def get_config_params(self):
        pass
        
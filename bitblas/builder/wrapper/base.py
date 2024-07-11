# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABC, abstractmethod


class BaseWrapper(ABC):

    @abstractmethod
    def wrap(self, *args, **kwargs):
        raise NotImplementedError

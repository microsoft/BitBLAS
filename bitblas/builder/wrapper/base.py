# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABC, abstractmethod


class BaseWrapper(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def wrap(self, *args, **kwargs):
        raise NotImplementedError

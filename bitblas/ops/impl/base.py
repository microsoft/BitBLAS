# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABC, abstractmethod


# TODO: Refactor all the tir script implementations to use this base class
# Abstract base class for TIR script emitters
class TIRScriptEmitter(ABC):

    @abstractmethod
    def emit(self):
        raise NotImplementedError


# Abstract base class for TIR script selectors
class TIRScriptSelector(ABC):

    @abstractmethod
    def select(self):
        raise NotImplementedError

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas.base.roller.hint import Hint
from abc import ABC, abstractmethod
from typing import Dict


# Base class for Tensor Layout Hints that defines the interface and common functionality for derived classes.
class BaseTLHint(ABC):

    # hint identifier
    hint_type: str = "base"

    # Constructor for the BaseTLHint class, takes variable arguments (*args and **kwargs) to allow flexibility.
    def __init__(self, *args, **kwargs):
        # Calls the superclass constructor (useful in complex inheritance hierarchies).
        super().__init__(*args, **kwargs)

    # Representation method to get a string representation of the object.
    # This method is not implemented here, so derived classes should provide their own implementation.
    def __repr__(self):
        raise NotImplementedError("method __repr__ is not implemented")

    # Class method to create an instance of BaseTLHint (or a derived class) from a Hint object.
    # This method needs to be implemented by subclasses.
    @classmethod
    def from_roller_hint(self, hint: Hint) -> 'BaseTLHint':
        raise NotImplementedError("method from_roller_hint is not implemented")

    # Abstract method to retrieve configuration parameters.
    # Derived classes must implement this method and return a dictionary of configuration parameters.
    @abstractmethod
    def get_config_params(self) -> Dict:
        pass

    # Allows the object to be accessed like a dictionary.
    # Retrieves a configuration parameter by key using the dictionary returned by get_config_params.
    def __getitem__(self, key):
        return self.get_config_params()[key]

    # Handles attempts to access non-existent attributes.
    # If the attribute is `get_config_params`, it returns the method itself.
    # Otherwise, raises an AttributeError if the attribute is not found.
    def __getattr__(self, item):
        # If the attribute is not found in the class, try to find it in the hint object
        if item == 'get_config_params':
            return self.get_config_params
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    # Allows the object to be iterated as if it were a dictionary.
    # Returns an iterator over the items (key-value pairs) in the configuration parameters.
    def __iter__(self):
        return iter(self.get_config_params().items())

    # Returns the keys of the configuration parameters as if the object were a dictionary.
    def keys(self):
        return self.get_config_params()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
import inspect
import pytest

# pytest.main() wrapper to allow running single test file
def main():
    test_file = inspect.getsourcefile(sys._getframe(1))
    sys.exit(pytest.main([test_file] + sys.argv[1:]))

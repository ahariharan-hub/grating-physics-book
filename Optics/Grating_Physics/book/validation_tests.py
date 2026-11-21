"""Locate optional private regressions without distributing their test sources."""
import os
from pathlib import Path


def private_test_directory(root):
    selected = os.environ.get('GRATING_PRIVATE_TESTS_ROOT')
    directory = Path(selected).resolve() if selected else Path(root) / 'supplement/tests'
    if selected and not directory.is_dir():
        raise RuntimeError('Configured private regression-test directory is missing.')
    if directory.is_dir():
        return directory
    print('Private regression suite is not distributed; standalone numerical/callback checks continue.')
    return None

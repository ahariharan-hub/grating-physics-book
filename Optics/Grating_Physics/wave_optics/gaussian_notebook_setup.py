"""Refresh owned teaching views when notebook source changes in a live kernel."""
from importlib import import_module, reload


def create_gaussian_explorer(previous=None):
    """Close the old instance, reload view dependencies, then create a fresh one.

    A regular ``from ... import`` can retain an older class in Jupyter's module
    cache. Reload only presentation/diagnostic modules; Gaussian physics stays
    untouched. This behaves like the notebook's existing reset-on-reexecution.
    """
    if previous is not None:
        previous.close()
    modules = (
        'gaussian_order_interactive_view',
        'gaussian_fringe_diagnostics',
        'gaussian_lesson_figures',
        'gaussian_order_beams_view',
    )
    for name in modules:
        module = reload(import_module('wave_optics.'+name))
    return module.GaussianOrderBeamsExplorer()

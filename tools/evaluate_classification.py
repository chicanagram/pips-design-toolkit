#!/usr/bin/env python3
"""Backward-compatible wrapper for the renamed sklearn evaluation entrypoint."""

from tools.evaluate_sklearn_classification import *  # noqa: F401,F403


if __name__ == "__main__":
    from runpy import run_module

    run_module("tools.evaluate_sklearn_classification", run_name="__main__")

"""Pytest fixtures + global setup for the Transoria suite."""

from __future__ import annotations

import os


# Suppress LLM IO logs during tests — the runner prints SEND/RECV lines
# to stderr by default (see ``transoria/llm/io_log.py``); under pytest
# this would flood the captured output and slow runs.
os.environ.setdefault("TRANSORIA_LLM_LOG", "off")

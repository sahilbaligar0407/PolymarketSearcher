"""Numbered SQL migrations applied in order by :func:`marketlab.storage.schema.run_migrations`.

This package holds no Python logic - it exists so the ``migrations`` directory is an
importable package and its ``*.sql`` files are packaged alongside the code.  Add new
migrations as ``NNN_description.sql`` with a strictly increasing, previously-unused
zero-padded number; never edit a migration that has already shipped.
"""

from __future__ import annotations

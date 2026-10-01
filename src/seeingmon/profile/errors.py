"""Errors that the profile package raises."""

from __future__ import annotations


class ProfileError(ValueError):
    """A profile is invalid or missing, or it lacks a value that the caller asked for.

    The message says what is wrong and where, in plain words, so you can show it to a user.
    """

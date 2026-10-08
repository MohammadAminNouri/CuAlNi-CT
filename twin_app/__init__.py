"""Standalone twin-family workbench package.

Imports are intentionally lightweight. Scientific backends are loaded only by
modules that actually run a calculation, which keeps pure input helpers and
static validation usable independently.
"""

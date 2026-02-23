"""Time formatting utilities for training progress tracking."""

from typing import Union


def format_time(seconds: Union[int, float]) -> str:
    """
    Convert time in seconds to human-readable format (hours, minutes, seconds).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string

    Example:
        >>> format_time(3665)
        '1 hours, 1 minutes, 5 seconds'
        >>> format_time(125)
        '0 hours, 2 minutes, 5 seconds'
        >>> format_time(45)
        '0 hours, 0 minutes, 45 seconds'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours} hours, {minutes} minutes, {secs} seconds"


def format_time_compact(seconds: Union[int, float]) -> str:
    """
    Convert time in seconds to compact format (HH:MM:SS).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string in HH:MM:SS format

    Example:
        >>> format_time_compact(3665)
        '01:01:05'
        >>> format_time_compact(125)
        '00:02:05'
        >>> format_time_compact(45)
        '00:00:45'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_eta(seconds_remaining: Union[int, float]) -> str:
    """
    Format estimated time remaining.

    Args:
        seconds_remaining: Estimated seconds remaining

    Returns:
        Formatted ETA string

    Example:
        >>> format_eta(3665)
        'ETA: 1h 1m 5s'
        >>> format_eta(125)
        'ETA: 2m 5s'
        >>> format_eta(45)
        'ETA: 45s'
    """
    hours = int(seconds_remaining // 3600)
    minutes = int((seconds_remaining % 3600) // 60)
    secs = int(seconds_remaining % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:  # Show seconds if it's the only unit
        parts.append(f"{secs}s")

    return "ETA: " + " ".join(parts)

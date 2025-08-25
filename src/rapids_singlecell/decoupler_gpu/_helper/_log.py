from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _log(message: str, level: str = "info", *, verbose: bool = False) -> None:
    """
    Log a message with a specified logging level.

    Parameters
    ----------
    message
        The message to log.
    level
        The logging level.
    verbose
        Whether to emit the log.
    """
    level = level.lower()
    if verbose:
        if level == "warn":
            logging.warning(message)
        elif level == "info":
            logging.info(message)

"""Rich-based logging utilities for mkm_solver."""

import inspect
import time
from datetime import datetime

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.progress import (
    Task as RichTask,
)
from rich.text import Text

_console: Console | None = None
_start_time: float | None = None


def _get_console() -> Console:
    global _console, _start_time
    if _console is None:
        _console = Console()
        _start_time = time.time()
    return _console


def _elapsed() -> str:
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    dt = time.time() - _start_time
    return f"{dt:.1f}s"


def _caller_module() -> str:
    frame = inspect.currentframe()
    if frame is None:
        return "unknown"
    try:
        caller = frame.f_back
        if caller is not None:
            caller = caller.f_back
        if caller is not None:
            caller = caller.f_back
        if caller is not None:
            return caller.f_globals.get("__name__", "unknown")
        return "unknown"
    finally:
        del frame


def get_logger() -> Console:
    return _get_console()


_LEVEL_STYLES = {
    "INFO": "blue",
    "SUCCESS": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "DEBUG": "dim",
}


class RateColumn(ProgressColumn):
    def render(self, task: RichTask) -> Text:
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("-- it/s", style="dim")
        if speed >= 1.0:
            return Text(f"{speed:.1f} it/s", style="magenta")
        return Text(f"{1.0 / speed:.1f} s/it", style="magenta")


class Logger:
    """Simple logger with Rich formatting.

    Output format (fixed-width prefix, message always starts at same column)::

        HH:MM:SS 0.1s INFO:source.module          message
    """

    _PREFIX_WIDTH = 35

    def __init__(self) -> None:
        self._console = _get_console()

    def _log(self, level: str, message: str) -> None:
        color = _LEVEL_STYLES[level]
        ts = datetime.now().strftime("%H:%M:%S")
        dt = _elapsed()
        src = _caller_module()
        tag = f"{level}:{src}"
        self._console.print(
            f"[dim]{ts} {dt:>6s}[/dim] [{color}]{tag:<{self._PREFIX_WIDTH}s}[/{color}] "
            f"[{color}]{message}[/{color}]"
        )

    def info(self, message: str) -> None:
        self._log("INFO", message)

    def success(self, message: str) -> None:
        self._log("SUCCESS", message)

    def warning(self, message: str) -> None:
        self._log("WARNING", message)

    def error(self, message: str) -> None:
        self._log("ERROR", message)

    def debug(self, message: str) -> None:
        self._log("DEBUG", message)

    def progress(self, description: str = "Processing", total: int | None = None) -> Progress:
        return Progress(
            SpinnerColumn(style="bold cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(
                bar_width=None,
                style="bar.back",
                complete_style="cyan",
                finished_style="green",
                pulse_style="cyan",
            ),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            RateColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(compact=True),
            console=self._console,
            expand=True,
        )


log = Logger()

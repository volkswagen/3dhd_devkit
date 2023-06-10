""" This module contains a method to log the console output to text files. """

import sys
from pathlib import Path
from typing import TextIO, Union


class _Logger(object):
    # Do not use this class directly, use set_logger instead for safe usage
    def __init__(self, std_stream: TextIO, log_file: Union[str, Path], overwrite=False):
        self.terminal = std_stream
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        mode = 'w' if overwrite else 'a+'
        self.file = open(self.log_file, mode, encoding='utf-8')

    def write(self, message: str):
        self.terminal.write(message)
        if not (message.endswith('it/s]') or message.endswith('s/it]')):  # do not log tqdm progress bars to text file
            self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()
        return self.terminal


def set_logger(log_file: Union[str, Path], overwrite: bool = False):
    """
    Sets up a logger which logs all console output (stdout, stderr) to a text file. In a multiprocessing setting, this
    function needs to be called once in every process for effective logging. Repeated calls of this function (inside the
    same process) do not have any additional effect, ensuring there is not more than one logger for every process.

    Args:
        log_file: full path to the text file (including file ending) to write the logs to
        overwrite: if log_file already exists, the logger will [True] delete all existing text or [False] keep the
                   text and start logging at the end of the file.

    """

    if log_file is None:
        raise TypeError("log_file path is None.")

    # reset logger if one already exists for another file (otherwise keep it)
    if isinstance(sys.stdout, _Logger) and sys.stdout.log_file != log_file:
        sys.stdout = sys.stdout.close()
    if isinstance(sys.stderr, _Logger) and sys.stderr.log_file != log_file:
        sys.stderr = sys.stderr.close()

    if not isinstance(sys.stdout, _Logger):
        sys.stdout = _Logger(sys.stdout, log_file, overwrite)
    if not isinstance(sys.stderr, _Logger):
        sys.stderr = _Logger(sys.stderr, log_file, overwrite)

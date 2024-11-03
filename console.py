import logging
import random
from time import sleep

from rich.console import Console
from rich.logging import RichHandler

from util import ASSETS_DIR

LOG = logging.getLogger(__name__)


ERROR_PREFIXES = [
    'ERROR',
    'CRIT',
    'ALERT',
]
WARN_PREFIXES = ['WARN']
DEBUG_PREFIXES = ['DEBUG']


def get_line(line: str) -> str:
    cleaned_line = ' - '.join(line.split(' - ')[1:])
    module_name = cleaned_line.split(' ')[0].replace(r'[', '').replace(']', '')
    # TODO(ew) rich won't render the bold text for some reason
    # module_name = f'[bold]\[{module_name}\][/bold]'
    module_name = f'[{module_name}]'
    return f'{module_name} - {" ".join(cleaned_line.split(" ")[1:])}'


def run_logs() -> None:
    FORMAT = "%(message)s"
    console = Console()
    logging.basicConfig(
        level="NOTSET",
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, tracebacks_show_locals=True)],
    )

    lines = []
    with open(ASSETS_DIR / 'logs.txt', 'r') as f:
        for line in f:
            lines.append(' '.join(line.split(' ')[2:]).strip())

    while True:
        line = random.choice(lines)
        match line:
            case line if any(p in line for p in ERROR_PREFIXES):
                LOG.error(get_line(line))
            case line if any(p in line for p in WARN_PREFIXES):
                LOG.warning(get_line(line))
            case line if any(p in line for p in DEBUG_PREFIXES):
                LOG.debug(get_line(line))
            case _:
                LOG.info(get_line(line))

        sleep(random.random())


if __name__ == '__main__':
    run_logs()

import datetime as dt
import logging
import os
from multiprocessing import Pool

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from util import OUTPUT_DIR


LOG = logging.getLogger(__name__)


COLORBAR_OUTPUT_DIR = OUTPUT_DIR / 'colorbars'


def make_inputs() -> pd.DataFrame:
    x = np.linspace(0, 4 * np.pi, 100)
    x2 = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    y2 = np.sin(x2)

    df = pd.DataFrame({'x': x, 'y': y, 'x2': x2, 'y2': y2})

    return df


def write_frame(df: pd.DataFrame, frame_id: int, title: str | None, output_dir: str) -> None:
    sns.barplot(
        df,
        x='x',
        y='y',
        hue='hue',
    )
    plt.savefig(f'{output_dir}/{frame_id}.png')


def generate_mov(name: str, test_pattern: bool = False, num_frames: int = 1000, length: int = 100, num_processes: int = 8, cleanup: bool = True) -> None:
    # setup outputs
    name = f'inplace-{name}'
    output_dir = COLORBAR_OUTPUT_DIR / f"raw__{name}_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    output_dir.mkdir()

    df = make_inputs()

    if test_pattern:
        sns.barplot(x='x', y='y', hue='y', data=df)
        plt.show()
        return

    LOG.info('Generating inputs')

    frames = []
    # TODO MAKE THE ANIMATION FOO
    # TODO STYLE THE ANIMATION FOO
    for _ in range(num_frames):
        frames.append(make_inputs())

    inputs = []
    for idx, frame in enumerate(frames):
        inputs.append((frame, idx, None, output_dir))

    LOG.info(f'Writing frames with {num_processes} threads')
    with Pool(processes=num_processes) as p:
        p.starmap(write_frame, inputs)

    LOG.info('Generating movie')
    mov_name = output_dir.name.split('__')[-1]
    ffmpeg.input(f'{output_dir}/*.png', pattern_type='glob', framerate=12).output(f'{DAY_TRADER_OUTPUT_DIR}/{mov_name}.mp4').run()

    if cleanup:
        LOG.info('Cleaning up raw frames directory')
        os.rmdir(output_dir)


if __name__ == '__main__':
    make_inputs()

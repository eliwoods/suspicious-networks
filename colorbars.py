import datetime as dt
import logging
import shutil
from multiprocessing import Pool

import ffmpeg
import numpy as np
import pandas as pd
import plotly.express as px

from util import OUTPUT_DIR


LOG = logging.getLogger(__name__)

COLORBAR_OUTPUT_DIR = OUTPUT_DIR / 'colorbars'


def make_inputs(samples: int = 100, phase: float = 0) -> pd.DataFrame:
    x = np.linspace(0, 4 * np.pi, samples)
    hue_x = np.linspace(0, 2 * np.pi, samples)

    y = np.sin(x + phase)
    hue_y = np.sin(hue_x + phase)

    df = pd.DataFrame({'x': x, 'y': y, 'hue': hue_y})

    return df


def write_frame(df: pd.DataFrame, frame_id: int, title: str | None, output_dir: str) -> None:
    fig = px.bar(
        df,
        x='x',
        y='y',
        color='hue',
        color_continuous_scale='Jet',
    )

    fig.update_layout(
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        showlegend=False,
        plot_bgcolor='black',
        paper_bgcolor='black',
        coloraxis_showscale=False,
    )

    fig.write_image(f'{output_dir}/{frame_id:06}.png')


def generate_mov(name: str, test_pattern: bool = False, num_frames: int = 1000, length: int = 100, num_processes: int = 8, cleanup: bool = True) -> None:
    # setup outputs
    name = f'inplace-{name}'
    output_dir = COLORBAR_OUTPUT_DIR / f"raw__{name}_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    output_dir.mkdir(parents=True)

    df = make_inputs()

    if test_pattern:
        fig = px.bar(
            df,
            x='x',
            y='y',
            color='hue',
            color_continuous_scale='Jet',
        )

        fig.update_layout(
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False),
            showlegend=False,
            plot_bgcolor='black',
            paper_bgcolor='black',
            coloraxis_showscale=False,
        )
        fig.show()
        return

    LOG.info('Generating inputs')

    phases = np.linspace(0, np.pi, num_frames)

    frames = []
    # TODO try a moire pattern with two waveforms colliding
    for phase in phases:
        frames.append(make_inputs(phase=phase))

    inputs = []
    for idx, frame in enumerate(frames):
        inputs.append((frame, idx, None, output_dir))

    LOG.info(f'Writing frames with {num_processes} threads')
    with Pool(processes=num_processes) as p:
        p.starmap(write_frame, inputs)

    LOG.info('Generating movie')
    mov_name = output_dir.name.split('__')[-1]
    ffmpeg.input(f'{output_dir}/*.png', pattern_type='glob', framerate=12).output(f'{COLORBAR_OUTPUT_DIR}/{mov_name}.mp4').run()

    if cleanup:
        LOG.info('Cleaning up raw frames directory')
        shutil.rmtree(output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    generate_mov(name='sin-sin', test_pattern=False, num_frames=50)

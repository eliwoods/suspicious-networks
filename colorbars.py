import datetime as dt
import logging
import shutil
from multiprocessing import Pool

import ffmpeg
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from util import OUTPUT_DIR


LOG = logging.getLogger(__name__)

COLORBAR_OUTPUT_DIR = OUTPUT_DIR / 'colorbars'


def make_inputs(samples: int = 100, phase: float = 0, hue: np.ndarray | None = None, sin_mod: float = 1) -> pd.DataFrame:
    x = np.linspace(0, 4 * np.pi, samples)
    y = np.sin(x / sin_mod + phase + np.pi) * np.sin(x + phase)

    if hue is None:
        hue_x = np.linspace(0, 2 * np.pi, samples)
        hue = np.sin(hue_x + phase)

    df = pd.DataFrame({'x': x, 'y': y, 'hue': hue})

    return df


def write_frame(df: tuple[pd.DataFrame, pd.DataFrame], frame_id: int, title: str | None, output_dir: str) -> None:
    fig = px.bar(
        df[0],
        x='x',
        y='y',
        color='hue',
        # color_continuous_scale='gray',
        color_continuous_scale='Plotly3',
        opacity=0.8,
    )

    fig.add_trace(
        go.Bar(
            x=df[1]['x'],
            y=df[1]['y'],
            marker=dict(
                color=df[1]['hue'],
                colorscale='Plotly3',
            ),
            opacity=0.8,
            # mode='markers',
        )
    )

    fig.update_layout(
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        showlegend=False,
        plot_bgcolor='black',
        paper_bgcolor='black',
        coloraxis_showscale=False,
        barmode='group',
    )

    fig.write_image(f'{output_dir}/{frame_id:06}.png')


def generate_mov(name: str, test_pattern: bool = False, num_frames: int = 1000, length: int = 100, num_processes: int = 8, cleanup: bool = True) -> None:
    # setup outputs
    name = f'inplace-{name}'
    output_dir = COLORBAR_OUTPUT_DIR / f"raw__{name}_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    output_dir.mkdir(parents=True)

    if test_pattern:
        df = make_inputs()

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

    phases_fwd = np.linspace(0, 4 * np.pi, num_frames)
    phases_bwd = np.linspace(5 * np.pi, np.pi, num_frames)

    hue_fwd = np.random.choice([0, 1], size=length, p=[0.5, 0.5])
    hue_bwd = np.random.choice([0, 1], size=length, p=[0.5, 0.5])

    # TODO Make these change over time?
    sin_mod_fwd = 2 * np.pi * np.random.randn() - np.pi
    sin_mod_bwd = 2 * np.pi * np.random.randn() - np.pi

    frames = []
    # TODO add another set of shapes going fwd and backward
    # TODO fix the figure size based on min/maxes
    for phase_fwd, phase_bwd in zip(phases_fwd, phases_bwd):
        # df_fwd = make_inputs(samples=length, phase=phase_fwd, hue=hue_fwd)
        # df_bwd = make_inputs(samples=length, phase=phase_bwd, hue=hue_bwd)
        df_fwd = make_inputs(samples=length, phase=phase_fwd, sin_mod=sin_mod_fwd)
        df_bwd = make_inputs(samples=length, phase=phase_bwd, sin_mod=sin_mod_bwd)
        hue_fwd = np.roll(hue_fwd, shift=1)
        hue_bwd = np.roll(hue_bwd, shift=-1)
        frames.append((df_fwd, df_bwd))

    inputs = []
    for idx, frame in enumerate(frames):
        inputs.append((frame, idx, None, output_dir))

    LOG.info(f'Writing frames with {num_processes} threads')
    with Pool(processes=num_processes) as p:
        p.starmap(write_frame, inputs)

    LOG.info('Generating movie')
    mov_name = output_dir.name.split('__')[-1]
    ffmpeg.input(f'{output_dir}/*.png', pattern_type='glob', framerate=24).output(f'{COLORBAR_OUTPUT_DIR}/{mov_name}.mp4').run()

    if cleanup:
        LOG.info('Cleaning up raw frames directory')
        shutil.rmtree(output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    generate_mov(name='sin-sin', test_pattern=False, num_frames=1000)

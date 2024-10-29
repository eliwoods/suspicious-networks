import datetime as dt
import logging
import shutil
from multiprocessing import Pool
from pathlib import Path

import ffmpeg
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from util import BASE_OUTPUT_DIR


LOG = logging.getLogger(__name__)

OUTPUT_DIR = BASE_OUTPUT_DIR / Path(__file__).stem


def make_fig(
    df: pd.DataFrame,
    names_col: str = 'names',
    values_col: str = 'values',
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    # df = df.sort_values(by=values_col, ascending=False)
    fig = px.pie(
        df,
        values=values_col,
        names=names_col,
        color=names_col,
        color_discrete_map=color_map,
        # category_orders=df[names_col].sort_values(ascending=flip_order),
    )

    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20, sort=False)

    fig.update_layout(
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        showlegend=False,
        plot_bgcolor='black',
        paper_bgcolor='black',
        coloraxis_showscale=False,
    )

    return fig


def write_frame(df: pd.DataFrame, frame_id: int, output_dir: str) -> None:
    fig = make_fig(df, color_map={'FUN': 'red', 'PROFIT': 'green'})
    fig.write_image(f'{output_dir}/{frame_id:06}.png')


def generate_mov(name: str, test_pattern: bool = False, num_frames: int = 1000, num_processes: int = 8, cleanup: bool = True) -> None:
    # setup outputs
    output_dir = OUTPUT_DIR / f"raw__{name}_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    output_dir.mkdir(parents=True)

    if test_pattern:
        df = pd.DataFrame({
            'names': ['FUN', 'PROFIT'],
            'values': [50, 50]
        })

        fig = make_fig(df, color_map={'FUN': 'green', 'PROFIT': 'red'})
        fig.show()
        return

    LOG.info('Generating inputs')

    fun_vals = np.concat([
        np.linspace(100, 0, num_frames // 2),
        np.linspace(0, 100, num_frames // 2),
    ])
    profit_vals = np.concat([
        np.linspace(0, 100, num_frames // 2),
        np.linspace(100, 0, num_frames // 2)
    ])

    frames = []
    for idx, (fun_val, profit_val) in enumerate(zip(fun_vals, profit_vals)):
        names = ['FUN', 'PROFIT'] if idx < num_frames // 2 else ['PROFIT', 'FUN']
        frames.append(pd.DataFrame({'values': [fun_val, profit_val], 'names': names}))

    inputs = []
    for idx, frame in enumerate(frames):
        inputs.append((frame, idx, output_dir))

    LOG.info(f'Writing frames with {num_processes} threads')
    with Pool(processes=num_processes) as p:
        p.starmap(write_frame, inputs)

    LOG.info('Generating movie')
    mov_name = output_dir.name.split('__')[-1]
    ffmpeg.input(f'{output_dir}/*.png', pattern_type='glob', framerate=24).output(f'{OUTPUT_DIR}/{mov_name}.mp4').run()

    if cleanup:
        LOG.info('Cleaning up raw frames directory')
        shutil.rmtree(output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    generate_mov(name='fun-profit-one-shot', test_pattern=False, num_frames=500)

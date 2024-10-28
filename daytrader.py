import logging
import datetime as dt
import shutil
from multiprocessing import Pool
from pathlib import Path

import ffmpeg

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf

from util import BASE_OUTPUT_DIR

LOG = logging.getLogger(__name__)


OUTPUT_DIR = BASE_OUTPUT_DIR / Path(__file__).stem


# Style utilities
def get_plot_style() -> dict:
    marketcolors = mpf.make_marketcolors(
        up='#39FF14',
        down='#000000',
        edge={'up': 'lime', 'down': 'red'},
        wick={'up': 'lime', 'down': 'red'},
    )
    style = mpf.make_mpf_style(
        base_mpf_style='mike',
        marketcolors=marketcolors,
        edgecolor='black',
        gridcolor='black',
        rc={
            'xtick.labelcolor': 'black',
            'ytick.labelcolor': 'black',
            'ytick.color': 'black',
            'xtick.color': 'black',
        }
    )

    return style


def test_style() -> None:
    df = yf.download('NVDA', start='2024-06-01')
    mpf.plot(
        df,
        type='candle',
        volume=False,
        style=get_plot_style(),
        warn_too_much_data=len(df) + 1,
        ylabel='',
    )


# Waveform functions
def triangle_wave(x: np.ndarray, period: float = 1, amplitude: np.ndarray | None = None, phase: float = 0) -> np.ndarray:
    if amplitude is None:
        amplitude = np.ones_like(x)

    return 2 * amplitude * np.abs(2 * ((x + phase) / period - np.floor((x + phase) / period + 0.5))) - amplitude


# Input generators
def make_inputs(length: int = 1000, test: bool = False, period: float | None = None, phase: float = 0, sigma_scale: float = 10, max_amplitude: float = 50) -> pd.DataFrame:
    x = np.linspace(start=0, stop=length - 1, num=length)

    amplitude = np.concat([
        np.linspace(start=1, stop=max_amplitude, num=length // 2),
        np.linspace(start=max_amplitude, stop=1, num=length // 2)
    ])

    if period is None:
        period = len(x) / 20

    y = triangle_wave(x, period=period, amplitude=amplitude, phase=phase)

    if test:
        plt.plot(x, y)
        plt.show()

    y_open = y + sigma_scale * np.random.randn(y.shape[0])
    y_close = y + sigma_scale * np.random.randn(y.shape[0])
    y_high = y + (amplitude/20 + sigma_scale * np.random.randn(y.shape[0]))
    y_low = y - (amplitude/20 + sigma_scale * np.random.randn(y.shape[0]))

    df = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=len(x), freq='D'),
        'Open': y_open,
        'Close': y_close,
        'High': y_high,
        'Low': y_low,
    })

    df.set_index('Date', inplace=True)

    return df


# Movie generation utilities
# TODO(ew) make decorator for setting up raw outputs and generating movies from frames so movie functions just define
#  their frame logic
def write_frame(df: pd.DataFrame, frame_id: int, ylim: tuple[float, float] | None, title: str | None, style: dict, output_dir: str) -> None:
    extra_kwargs = {}
    if ylim is not None:
        extra_kwargs['ylim'] = ylim

    if title is not None:
        extra_kwargs['title'] = title

    mpf.plot(
        df,
        type='candle',
        volume=False,
        style=style,
        savefig=f'{output_dir}/{frame_id:06}.png',
        warn_too_much_data=len(df) + 1,
        ylabel='',
        **extra_kwargs
    )


# Movie functions
def generate_sliding_mov(name: str, test_pattern: bool = False, length: int = 1000, num_processes: int = 8) -> None:
    """
    This animation acts a sliding window on a fixed pattern.
    :param name: Name prefix of the movie
    :param test_pattern: If true, just generates a single still of the whole pattern for testing
    :param length: Number of samples in the pattern
    :param num_processes: Number of processes when writing frames
    :return: None
    """
    # setup outputs
    name = f'sliding-{name}'
    output_dir = OUTPUT_DIR / f"raw__{name}_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    output_dir.mkdir()

    # Setup styles
    df = make_inputs(test=test_pattern, length=length)

    style = get_plot_style()

    if test_pattern:
        mpf.plot(
            df,
            type='candle',
            volume=False,
            style=style,
            warn_too_much_data=len(df) + 1,
            ylabel='',
        )
        return

    LOG.info('Generating inputs')
    size = 100
    start = 0
    stop = size
    step = 1
    inputs = []
    ylim = (df['Low'].min() - df['Low'].min() * 0.1, df['High'].max() + df['High'].max() * 0.1)
    while stop < len(df):
        if start != 0 and start % (length // 10) == 0:
            LOG.info(f'At frame {start} of {len(df) - size}')
        # frame_df = df[start:stop]
        frame_df = df[start:stop]
        inputs.append((frame_df, start, ylim, None, style, output_dir))
        start += step
        stop += step

    LOG.info(f'Writing frames with {num_processes} threads')
    with Pool(processes=num_processes) as p:
        p.starmap(write_frame, inputs)

    LOG.info('Generating movie')
    mov_name = output_dir.name.split('__')[-1]
    ffmpeg.input(f'{output_dir}/*.png', pattern_type='glob', framerate=45).output(f'{OUTPUT_DIR}/{mov_name}.mp4').run()


def generate_inplace_mov(name: str, test_pattern: bool = False, num_frames: int = 1000, length: int = 100, num_processes: int = 8, cleanup: bool = True) -> None:
    """
    This animation displays an entire pattern which can then be iterated on. Naturally the random upstreap processes
    will add noise, but further pattern modulation can occur
    :param name: Name prefix of the movie
    :param test_pattern: If true, just generates a single still of the whole pattern for testing
    :param num_frames: Number of frames in the animation
    :param length: Number of samples in the pattern
    :param num_processes: Number of processes when writing frames
    :return: None
    """
    # setup outputs
    name = f'inplace-{name}'
    output_dir = OUTPUT_DIR / f"raw__{name}_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    output_dir.mkdir()

    # Setup data and styles
    style = get_plot_style()

    if test_pattern:
        df = make_inputs(test=test_pattern, length=length)
        mpf.plot(
            df,
            type='candle',
            volume=False,
            style=style,
            warn_too_much_data=len(df) + 1,
            ylabel='',
        )
        return

    LOG.info('Generating inputs')
    samples = np.concat([
        np.linspace(10, length, num_frames // 2, dtype=int),
        np.linspace(length, 10, num_frames // 2, dtype=int),
    ])
    even_samples = samples + (samples % 2)

    periods = np.concat([
        np.linspace(length / 30, length / 20, num_frames // 2),
        np.linspace(length / 20, length / 30, num_frames // 2),
    ])

    phases = np.linspace(100, 200, num_frames)

    amplitudes = np.concat([
        np.linspace(10, 60, num_frames // 2),
        np.linspace(60, 10, num_frames // 2),
    ])

    frames = []
    # See the following for annotated sample lengths
    # /Users/eli/eliwoods/suspicious-networks/output/good-sample-reference-inplace-tri-wavelet_20241019T160221.mp4
    for _length, period, phase, amplitude in zip(even_samples, periods, phases, amplitudes):
        frames.append(make_inputs(length=58, phase=phase, sigma_scale=1.5, max_amplitude=amplitude))

    # Get ylims to standardize plots
    ymin = min([f['Low'].min() for f in frames])
    ymax = max([f['High'].max() for f in frames])
    ylim = (ymin - ymin*0.1, ymax + ymax*0.1)

    inputs = []
    for idx, frame in enumerate(frames):
        # title = f'num_samples: {[idx]} | phase: {phases[idx]:.2f} | period: {periods[idx]:.2f}'
        title = None
        inputs.append((frame, idx, ylim, title, style, output_dir))

    LOG.info(f'Writing frames with {num_processes} threads')
    with Pool(processes=num_processes) as p:
        p.starmap(write_frame, inputs)

    LOG.info('Generating movie')
    mov_name = output_dir.name.split('__')[-1]
    ffmpeg.input(f'{output_dir}/*.png', pattern_type='glob', framerate=12).output(f'{OUTPUT_DIR}/{mov_name}.mp4').run()

    if cleanup:
        LOG.info('Cleaning up raw frames directory')
        shutil.rmtree(output_dir)


if __name__ == '__main__':
    from pathlib import Path
    logging.basicConfig(level=logging.INFO)
    # generate_inplace_mov('tri-wavelet', test_pattern=False)
    print(Path(__file__).stem)

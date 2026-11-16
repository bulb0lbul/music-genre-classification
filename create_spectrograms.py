import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from trim_audio import load_trimmed_audio

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


def create_mel_spectrogram(audio, sample_rate: int):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0, center=True)
    if not np.isfinite(spectrogram).all() or not np.any(spectrogram > 0):
        raise ValueError("Мел-спектрограмма содержит некорректные значения или нулевую мощность.")
    return spectrogram


def create_spectrogram_example(data_dir: Path, track_id: int | None = None):
    tracks = pd.read_csv(data_dir / "processed/tracks_trimmed.csv")
    selected = tracks if track_id is None else tracks.loc[tracks["track_id"].eq(track_id)]
    if selected.empty:
        raise ValueError("Запись не найдена в tracks_trimmed.csv.")
    row = selected.iloc[0]
    audio, sample_rate = load_trimmed_audio(data_dir / row.audio_path, int(row.frames))
    spectrogram = create_mel_spectrogram(audio, sample_rate)
    return row, audio, sample_rate, spectrogram


def main() -> None:
    parser = argparse.ArgumentParser(description="Рассчитать мел-спектрограмму одной записи очищенного набора.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--track-id", type=int, help="ID записи; по умолчанию первая запись из tracks_trimmed.csv.")
    parser.add_argument("--power-max", type=float, help="Верхняя граница цветовой шкалы; по умолчанию 99-й перцентиль мощности.")
    parser.add_argument("--no-plots", action="store_true", help="Вывести параметры без построения графика.")
    args = parser.parse_args()
    try:
        data_dir = args.data_dir.expanduser().resolve()
        row, audio, sample_rate, spectrogram = create_spectrogram_example(data_dir, args.track_id)
        print(f"Запись: {int(row.track_id):06d} · {row.genre}")
        print(f"Аудио: {audio.shape}; {audio.dtype}; {sample_rate} Гц; {len(audio) / sample_rate:.8f} с")
        print(f"Параметры: {N_MELS} мел-полос; окно {N_FFT}; шаг {HOP_LENGTH} отсчётов")
        print(f"Мел-спектрограмма мощности: {spectrogram.shape}; {spectrogram.dtype}")
        print(f"Диапазон мощности: {spectrogram.min():.6g}–{spectrogram.max():.6g}")
        if not args.no_plots:
            from plot_data import plot_mel_spectrogram, plot_mel_scale_comparison
            path = Path(__file__).resolve().parent / "plots" / f"mel_spectrogram_{int(row.track_id):06d}.png"
            plot_mel_spectrogram(spectrogram, sample_rate, HOP_LENGTH, path, f"{int(row.track_id):06d} · {row.genre}", power_max=args.power_max)
            comparison_path = path.with_name(f"mel_scale_comparison_{int(row.track_id):06d}.png")
            plot_mel_scale_comparison(spectrogram, sample_rate, HOP_LENGTH, comparison_path, f"{int(row.track_id):06d} · {row.genre}")
            print(f"График: {path}")
            print(f"Сравнение шкал: {comparison_path}")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ошибка построения мел-спектрограммы: {error}\n")


if __name__ == "__main__":
    main()

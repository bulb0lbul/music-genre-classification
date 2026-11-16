import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from prepare_data import save_csv


def load_audio(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Аудиофайл не найден: {path}")
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if len(audio) == 0:
        raise ValueError("Аудиофайл не содержит отсчётов.")
    return audio, sample_rate


def to_mono(audio):
    return audio.mean(axis=1)


def load_mono_audio(path: Path):
    audio, sample_rate = load_audio(path)
    return to_mono(audio), sample_rate


def rms_envelope(audio, sample_rate: int):
    size = max(1, sample_rate // 10)
    count = len(audio) // size
    if not count:
        return np.array([0.0]), np.array([np.sqrt(np.mean(np.square(audio, dtype=np.float64)))])
    values = audio[:count * size].reshape(count, -1).astype(np.float64)
    return (np.arange(count) + 0.5) * size / sample_rate, np.sqrt(np.mean(values * values, axis=1))


def prepare_mono_dataset(data_dir: Path) -> pd.DataFrame:
    tracks = pd.read_csv(data_dir / "processed/tracks_clean.csv")
    if tracks.empty:
        raise ValueError("Очищенный набор пуст.")
    for path in tracks["audio_path"]:
        if not (data_dir / path).is_file():
            raise FileNotFoundError(f"Аудиофайл не найден: {path}")
    processed = tracks[["track_id", "genre", "label", "split", "artist_id", "audio_path", "sample_rate", "channels", "frames", "duration_seconds"]].copy()
    for column in ("sample_rate", "channels", "frames", "duration_seconds"):
        processed[f"source_{column}"] = processed[column]
    processed["channels"] = 1
    path = data_dir / "processed/tracks_mono.csv"
    save_csv(path, processed)
    print(f"Подготовлено для сведения в моно при загрузке: {len(processed)} записей; WAV-копии не создаются.", flush=True)
    print(f"Таблица: {path}", flush=True)
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Перевести одну запись или очищенный набор в моно.")
    parser.add_argument("audio_path", type=Path, nargs="?")
    parser.add_argument("--all", action="store_true", help="Подготовить таблицу для сведения очищенного набора в моно при загрузке.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--no-plots", action="store_true", help="Пропустить графики; в пайплайне они строятся отдельным этапом.")
    args = parser.parse_args()
    if args.all and args.audio_path:
        parser.error("Укажите путь к одной записи или --all.")
    if not args.all and not args.audio_path:
        parser.error("Укажите путь к записи или --all.")
    try:
        if args.all:
            data_dir = args.data_dir.expanduser().resolve()
            processed = prepare_mono_dataset(data_dir)
            if not args.no_plots:
                from plot_data import plot_mono_dataset, plot_waveforms, create_example_plots
                original = pd.read_csv(data_dir / "processed/tracks_clean.csv")
                plots_dir = Path(__file__).resolve().parent / "plots"
                plot_mono_dataset(original, processed, data_dir, plots_dir / "mono")
                plot_waveforms(processed, data_dir, plots_dir / "random_waveforms.png")
                create_example_plots(data_dir, plots_dir)
        else:
            path = args.audio_path.expanduser().resolve()
            audio, sample_rate = load_audio(path)
            mono = to_mono(audio)
            print(f"Файл: {path}")
            print(f"Частота дискретизации: {sample_rate} Гц")
            print(f"Каналов до обработки: {audio.shape[1]}; после: 1")
            print(f"Длительность: {len(mono) / sample_rate:.4f} с")
            print(f"Форма массива: {mono.shape}")
            if not args.no_plots:
                from plot_data import plot_mono_example
                figure = Path(__file__).resolve().parent / "plots/mono" / f"track_{path.stem}.png"
                plot_mono_example(audio, mono, sample_rate, figure, f"{path.stem} · {audio.shape[1]} → 1 канал · {sample_rate} Гц")
                print(f"График: {figure}")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ошибка обработки аудио: {error}\n")


if __name__ == "__main__":
    main()

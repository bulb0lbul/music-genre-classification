import argparse
from pathlib import Path

import pandas as pd
from prepare_data import save_csv
from resample_audio import TARGET_SAMPLE_RATE, load_resampled_audio


def trim_signal(audio, frames: int):
    if frames <= 0 or len(audio) < frames:
        raise ValueError(f"В записи недостаточно отсчётов для фрагмента длины {frames}.")
    if audio.ndim != 1:
        raise ValueError("Для обрезки нужен одномерный моносигнал.")
    return audio[:frames]


def load_trimmed_audio(path: Path, frames: int):
    audio, sample_rate = load_resampled_audio(path)
    return trim_signal(audio, frames), sample_rate


def select_check_examples(tracks: pd.DataFrame) -> pd.DataFrame:
    formats = tracks.drop_duplicates(["source_sample_rate", "source_channels"])
    lengths = tracks.loc[[tracks["frames_before_trim"].idxmin(), tracks["frames_before_trim"].idxmax()]]
    return pd.concat([formats, lengths]).drop_duplicates("track_id").sort_values(["source_sample_rate", "source_channels", "track_id"])


def trim_dataset(data_dir: Path) -> pd.DataFrame:
    tracks = pd.read_csv(data_dir / "processed/tracks_resampled.csv")
    if tracks.empty or not tracks["sample_rate"].eq(TARGET_SAMPLE_RATE).all() or not tracks["channels"].eq(1).all():
        raise ValueError(f"Для обрезки нужна таблица моносигналов с целевой частотой {TARGET_SAMPLE_RATE} Гц.")
    target_frames = int(tracks["frames"].min())
    if target_frames <= 0:
        raise ValueError("В таблице есть пустая запись.")
    for path in tracks["audio_path"]:
        if not (data_dir / path).is_file():
            raise FileNotFoundError(f"Аудиофайл не найден: {path}")
    processed = tracks.copy()
    processed["frames_before_trim"] = processed["frames"]
    processed["duration_before_trim_seconds"] = processed["duration_seconds"]
    processed["trim_start_frame"] = 0
    processed["trim_end_frame"] = target_frames
    processed["cropped_frames"] = processed["frames"] - target_frames
    processed["was_trimmed"] = processed["cropped_frames"].gt(0)
    processed["frames"] = target_frames
    processed["duration_seconds"] = target_frames / TARGET_SAMPLE_RATE
    path = data_dir / "processed/tracks_trimmed.csv"
    save_csv(path, processed)
    print(f"Длина при загрузке: {target_frames} отсчётов; {target_frames / TARGET_SAMPLE_RATE:.8f} с; частота {TARGET_SAMPLE_RATE} Гц", flush=True)
    print(f"Фрагментов с обрезкой конца: {processed['was_trimmed'].sum()}; без обрезки: {(~processed['was_trimmed']).sum()}", flush=True)
    print(f"Максимально удаляется с конца: {processed['cropped_frames'].max() / TARGET_SAMPLE_RATE * 1000:.2f} мс", flush=True)
    print(f"Таблица: {path}. Для чтения фрагментов используется load_trimmed_audio.", flush=True)
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовить фрагменты минимальной длины с обрезкой при загрузке, без создания WAV-копий.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--no-plots", action="store_true", help="Пропустить графики; в пайплайне они строятся отдельным этапом.")
    args = parser.parse_args()
    try:
        data_dir = args.data_dir.expanduser().resolve()
        processed = trim_dataset(data_dir)
        if not args.no_plots:
            from plot_data import plot_trimming
            plot_trimming(processed, data_dir, Path(__file__).resolve().parent / "plots/trimming_examples.png")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ошибка подготовки фрагментов: {error}\n")


if __name__ == "__main__":
    main()

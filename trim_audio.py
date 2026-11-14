import argparse
from pathlib import Path

import pandas as pd
from resample_audio import TARGET_SAMPLE_RATE, load_resampled_audio


def load_trimmed_audio(path: Path, frames: int):
    audio, sample_rate = load_resampled_audio(path)
    if frames <= 0 or len(audio) < frames:
        raise ValueError(f"В записи недостаточно отсчётов для фрагмента длины {frames}: {path}")
    audio = audio[:frames]
    if sample_rate != TARGET_SAMPLE_RATE or audio.shape != (frames,):
        raise ValueError(f"Нужен монофрагмент из {frames} отсчётов с частотой 44 100 Гц: {path}")
    return audio, sample_rate


def trim_dataset(data_dir: Path, make_plots: bool = True) -> pd.DataFrame:
    tracks = pd.read_csv(data_dir / "processed/tracks_resampled.csv")
    if tracks.empty or not tracks["sample_rate"].eq(44100).all() or not tracks["channels"].eq(1).all():
        raise ValueError("Для обрезки нужна таблица моносигналов с целевой частотой 44 100 Гц.")
    target_frames = int(tracks["frames"].min())
    if target_frames <= 0:
        raise ValueError("В таблице есть пустая запись.")
    records = []
    for number, row in enumerate(tracks.itertuples(), 1):
        source = data_dir / row.audio_path
        if not source.is_file():
            raise FileNotFoundError(f"Аудиофайл не найден: {source}")
        record = row._asdict()
        record.pop("Index")
        record.update(pre_trim_audio_path=row.audio_path, frames_before_trim=row.frames, duration_before_trim_seconds=row.duration_seconds, trim_start_frame=0, trim_end_frame=target_frames, cropped_frames=row.frames - target_frames, was_trimmed=row.frames > target_frames, frames=target_frames, duration_seconds=target_frames / row.sample_rate)
        records.append(record)
        if number % 1000 == 0 or number == len(tracks):
            print(f"Подготовлены фрагменты: {number}/{len(tracks)}", flush=True)
    processed = pd.DataFrame(records)
    path = data_dir / "processed/tracks_trimmed.csv"
    temporary = path.with_suffix(".csv.part")
    processed.to_csv(temporary, index=False)
    temporary.replace(path)
    print(f"Длина при загрузке: {target_frames} отсчётов; {target_frames / 44100:.8f} с; частота 44 100 Гц", flush=True)
    print(f"Фрагментов с обрезкой конца: {processed['was_trimmed'].sum()}; без обрезки: {(~processed['was_trimmed']).sum()}", flush=True)
    print(f"Максимально удаляется с конца: {processed['cropped_frames'].max() / 44100 * 1000:.2f} мс", flush=True)
    print(f"Таблица: {path}. Для чтения фрагментов используется load_trimmed_audio.", flush=True)
    if make_plots:
        from plot_data import plot_trimming
        plot_trimming(processed, data_dir, Path(__file__).resolve().parent / "plots/trimming_examples.png")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовить фрагменты минимальной длины с обрезкой при загрузке, без создания WAV-копий.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--no-plots", action="store_true", help="Пропустить графики; в пайплайне они строятся отдельным этапом.")
    args = parser.parse_args()
    try:
        trim_dataset(args.data_dir.expanduser().resolve(), make_plots=not args.no_plots)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ошибка подготовки фрагментов: {error}\n")


if __name__ == "__main__":
    main()

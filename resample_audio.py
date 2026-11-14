import argparse
from pathlib import Path

import librosa
import pandas as pd
from process_audio import load_mono_audio


TARGET_SAMPLE_RATE = 44100


def resample_signal(audio, sample_rate: int):
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio, TARGET_SAMPLE_RATE
    return librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE, res_type="soxr_hq"), TARGET_SAMPLE_RATE


def load_resampled_audio(path: Path):
    audio, sample_rate = load_mono_audio(path)
    return resample_signal(audio, sample_rate)


def resample_dataset(data_dir: Path, make_plots: bool = True) -> pd.DataFrame:
    tracks = pd.read_csv(data_dir / "processed/tracks_mono.csv")
    if tracks.empty or not tracks["channels"].eq(1).all() or not tracks[["sample_rate", "frames"]].gt(0).all().all():
        raise ValueError("Нужна таблица непустых записей для сведения в моно.")
    records = []
    for row in tracks.itertuples():
        changed = row.sample_rate != TARGET_SAMPLE_RATE
        expected_frames = (row.frames * TARGET_SAMPLE_RATE + row.sample_rate - 1) // row.sample_rate
        duration = expected_frames / TARGET_SAMPLE_RATE
        record = row._asdict()
        record.pop("Index")
        record.update(source_sample_rate=row.sample_rate, source_frames=row.frames, source_duration_seconds=row.duration_seconds, was_resampled=changed, resample_method="soxr_hq" if changed else "unchanged", sample_rate=TARGET_SAMPLE_RATE, frames=expected_frames, duration_seconds=duration, duration_delta_seconds=duration - row.duration_seconds)
        records.append(record)
    processed = pd.DataFrame(records)
    path = data_dir / "processed/tracks_resampled.csv"
    temporary = path.with_suffix(".csv.part")
    processed.to_csv(temporary, index=False)
    temporary.replace(path)
    print(f"Ресемплинг при загрузке: {processed['was_resampled'].sum()}; исходная частота уже подходит: {(~processed['was_resampled']).sum()}", flush=True)
    print(f"Таблица: {path}. WAV-копии не создаются.", flush=True)
    if make_plots:
        from plot_data import plot_resampling
        plot_resampling(processed, data_dir, Path(__file__).resolve().parent / "plots")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовить ресемплинг до 44 100 Гц при загрузке с сохранением исходных частот.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--no-plots", action="store_true", help="Пропустить графики; в пайплайне они строятся отдельным этапом.")
    args = parser.parse_args()
    try:
        resample_dataset(args.data_dir.expanduser().resolve(), make_plots=not args.no_plots)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ошибка ресемплинга: {error}\n")


if __name__ == "__main__":
    main()

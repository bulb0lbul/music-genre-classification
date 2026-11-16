import argparse
from pathlib import Path

import librosa
import pandas as pd
from prepare_data import save_csv
from process_audio import load_mono_audio


TARGET_SAMPLE_RATE = 44100


def resample_signal(audio, sample_rate: int):
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio, TARGET_SAMPLE_RATE
    return librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE, res_type="soxr_hq"), TARGET_SAMPLE_RATE


def load_resampled_audio(path: Path):
    audio, sample_rate = load_mono_audio(path)
    return resample_signal(audio, sample_rate)


def resample_dataset(data_dir: Path) -> pd.DataFrame:
    tracks = pd.read_csv(data_dir / "processed/tracks_mono.csv")
    if tracks.empty or not tracks["channels"].eq(1).all() or not tracks[["source_sample_rate", "source_frames"]].gt(0).all().all():
        raise ValueError("Нужна таблица непустых записей для сведения в моно.")
    processed = tracks.copy()
    processed["was_resampled"] = processed["source_sample_rate"].ne(TARGET_SAMPLE_RATE)
    processed["resample_method"] = processed["was_resampled"].map({True: "soxr_hq", False: "unchanged"})
    processed["sample_rate"] = TARGET_SAMPLE_RATE
    processed["frames"] = (processed["source_frames"] * TARGET_SAMPLE_RATE + processed["source_sample_rate"] - 1) // processed["source_sample_rate"]
    processed["duration_seconds"] = processed["frames"] / TARGET_SAMPLE_RATE
    processed["duration_delta_seconds"] = processed["duration_seconds"] - processed["source_duration_seconds"]
    path = data_dir / "processed/tracks_resampled.csv"
    save_csv(path, processed)
    print(f"Ресемплинг при загрузке: {processed['was_resampled'].sum()}; исходная частота уже подходит: {(~processed['was_resampled']).sum()}", flush=True)
    print(f"Таблица: {path}. WAV-копии не создаются.", flush=True)
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовить ресемплинг до 44 100 Гц при загрузке с сохранением исходных частот.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--no-plots", action="store_true", help="Пропустить графики; в пайплайне они строятся отдельным этапом.")
    args = parser.parse_args()
    try:
        data_dir = args.data_dir.expanduser().resolve()
        processed = resample_dataset(data_dir)
        if not args.no_plots:
            from plot_data import plot_resampling
            plot_resampling(processed, data_dir, Path(__file__).resolve().parent / "plots")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ошибка ресемплинга: {error}\n")


if __name__ == "__main__":
    main()

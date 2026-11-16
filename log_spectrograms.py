import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from create_spectrograms import HOP_LENGTH, create_mel_spectrogram
from prepare_data import save_csv
from trim_audio import load_trimmed_audio

REFERENCE_POWER = 1.0
MIN_POWER = 1e-10
TOP_DB = 80.0


def to_log_mel(spectrogram):
    if spectrogram.ndim != 2 or not spectrogram.size or not np.isfinite(spectrogram).all() or np.any(spectrogram < 0) or not np.any(spectrogram > 0):
        raise ValueError("Для перевода в дБ нужна непустая матрица конечной неотрицательной мощности.")
    return librosa.power_to_db(spectrogram, ref=REFERENCE_POWER, amin=MIN_POWER, top_db=TOP_DB).astype(np.float32, copy=False)


def prepare_log_examples(data_dir: Path, track_id: int | None = None):
    tracks = pd.read_csv(data_dir / "processed/tracks_trimmed.csv")
    if track_id is None:
        selected = tracks.loc[tracks["split"].eq("train")].groupby("genre", sort=True).sample(n=1, random_state=42).sort_values("genre")
    else:
        selected = tracks.loc[tracks["track_id"].eq(track_id)]
    if selected.empty:
        raise ValueError("Нет записей для построения логарифмических спектрограмм.")
    examples = []
    records = []
    for row in selected.itertuples(index=False):
        audio, sample_rate = load_trimmed_audio(data_dir / row.audio_path, int(row.frames))
        power = create_mel_spectrogram(audio, sample_rate)
        log_mel = to_log_mel(power)
        if examples and (log_mel.shape != examples[0]["spectrogram"].shape or sample_rate != examples[0]["sample_rate"]):
            raise ValueError("Формы спектрограмм или частоты дискретизации примеров различаются.")
        unbounded_db = librosa.power_to_db(power, ref=REFERENCE_POWER, amin=MIN_POWER, top_db=None)
        floored_percent = float(np.mean(log_mel > unbounded_db)) * 100
        examples.append({"track_id": int(row.track_id), "genre": row.genre, "split": row.split, "sample_rate": sample_rate, "spectrogram": log_mel})
        records.append({"track_id": int(row.track_id), "genre": row.genre, "split": row.split, "source_sample_rate": int(row.source_sample_rate), "was_resampled": bool(row.was_resampled), "mel_bands": log_mel.shape[0], "time_frames": log_mel.shape[1], "dtype": str(log_mel.dtype), "min_db": float(log_mel.min()), "max_db": float(log_mel.max()), "mean_db": float(log_mel.mean()), "std_db": float(log_mel.std()), "cutoff_db": float(log_mel.max()) - TOP_DB, "floored_percent": floored_percent})
    summary = pd.DataFrame(records)
    report_dir = Path(__file__).resolve().parent / "reports/log_spectrograms"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / ("examples.csv" if track_id is None else f"track_{track_id:06d}.csv")
    save_csv(report_path, summary)
    print(f"Преобразование: мощность → дБ; ref={REFERENCE_POWER}; amin={MIN_POWER:g}; top_db={TOP_DB:g}", flush=True)
    print(f"Форма каждого массива: {examples[0]['spectrogram'].shape}; float32", flush=True)
    print(summary[["track_id", "genre", "split", "min_db", "max_db", "floored_percent"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"), flush=True)
    print(f"Таблица проверки: {report_path}", flush=True)
    return examples, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Перевести мел-спектрограммы в дБ и проверить по одному примеру каждого жанра из train.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--track-id", type=int, help="Проверить одну выбранную запись вместо примеров восьми жанров.")
    parser.add_argument("--no-plots", action="store_true", help="Вывести результаты и сохранить CSV без графиков.")
    args = parser.parse_args()
    try:
        examples, summary = prepare_log_examples(args.data_dir.expanduser().resolve(), args.track_id)
        if not args.no_plots:
            from plot_data import plot_log_mel_spectrogram, plot_log_mel_examples, plot_log_mel_diagnostics
            output_dir = Path(__file__).resolve().parent / "plots"
            if len(examples) == 1:
                example = examples[0]
                path = output_dir / f"mel_log_spectrogram_{example['track_id']:06d}.png"
                plot_log_mel_spectrogram(example["spectrogram"], example["sample_rate"], HOP_LENGTH, path, f"{example['track_id']:06d} · {example['genre']}")
            else:
                plot_log_mel_examples(examples, output_dir / "log_mel_examples.png")
                plot_log_mel_diagnostics(examples, summary, output_dir / "log_mel_diagnostics.png")
            print(f"Графики: {output_dir}", flush=True)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ошибка логарифмического преобразования: {error}\n")


if __name__ == "__main__":
    main()

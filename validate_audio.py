import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib
import math
import multiprocessing
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import soundfile as sf

from prepare_data import GENRES, save_csv, save_json


@contextmanager
def decoder_messages():
    with tempfile.TemporaryFile() as stream:
        saved = os.dup(2)
        try:
            os.dup2(stream.fileno(), 2)
            yield stream
        finally:
            os.dup2(saved, 2)
            os.close(saved)


def inspect_audio(path: Path, min_duration: float) -> dict:
    result = dict(sample_rate=None, channels=None, frames=None, duration_seconds=None, pcm_sha256="", decoder_log="", exclusion_reasons="")
    reasons = []
    if not path.is_file() or path.stat().st_size == 0:
        result["exclusion_reasons"] = "missing_or_empty_audio"
        return result
    frames, peak = 0, 0.0
    digest = hashlib.sha256()
    with decoder_messages() as diagnostics:
        try:
            with sf.SoundFile(path) as source:
                sr, channels = source.samplerate, source.channels
                result.update(sample_rate=sr, channels=channels)
                digest.update(f"float32-le:{sr}:{channels}:".encode())
                for block in source.blocks(blocksize=65536, dtype="float32", always_2d=True):
                    if not np.isfinite(block).all():
                        reasons.append("nonfinite_samples")
                        break
                    frames += len(block)
                    peak = max(peak, float(np.max(np.abs(block))))
                    digest.update(block.astype("<f4", copy=False).tobytes())
                result.update(frames=frames, duration_seconds=frames / sr)
        except (RuntimeError, ValueError, OSError) as error:
            reasons.append("decode_error")
            result["decode_error"] = str(error)
        diagnostics.seek(0)
        result["decoder_log"] = diagnostics.read(8192).decode("utf-8", errors="replace")
    log = result["decoder_log"].lower()
    if any(text in log for text in ("error:", "giving up", "dequantization failed")):
        reasons.append("decoder_reported_error")
    if "decode_error" not in reasons:
        if frames == 0:
            reasons.append("empty_audio")
        elif result["duration_seconds"] < min_duration:
            reasons.append("too_short")
        if peak == 0 and frames:
            reasons.append("silent_audio")
        if not reasons:
            result["pcm_sha256"] = digest.hexdigest()
    result["exclusion_reasons"] = ";".join(sorted(set(reasons)))
    return result


def audit_track(task: tuple) -> dict:
    record, data_dir, min_duration = task
    if not record["metadata_valid"]:
        return {**record, "exclusion_reasons": record["metadata_issues"]}
    return {**record, **inspect_audio(Path(data_dir) / record["audio_path"], min_duration)}


def remove_duplicates(audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    duplicates = []
    valid = audit.loc[audit["exclusion_reasons"].eq("")]
    for _, group in valid.groupby("pcm_sha256"):
        if len(group) < 2:
            continue
        reasons = []
        if group["genre"].nunique() > 1:
            reasons.append("duplicate_label_conflict")
        if group["split"].nunique() > 1:
            reasons.append("duplicate_across_splits")
        keeper = group["track_id"].idxmin()
        for index, row in group.iterrows():
            reason = ";".join(reasons) if reasons else ("duplicate_same_split" if index != keeper else "")
            audit.at[index, "exclusion_reasons"] = reason
            duplicates.append(dict(duplicate_group=int(group["track_id"].min()), track_id=int(row.track_id), genre=row.genre, split=row.split, reason=reason, action="exclude" if reason else "keep"))
    columns = ["duplicate_group", "track_id", "genre", "split", "reason", "action"]
    return audit, pd.DataFrame(duplicates, columns=columns)


def validate_dataset(data_dir: Path, manifest: Path, output_dir: Path, min_duration: float, workers: int) -> dict:
    table = pd.read_csv(manifest, keep_default_na=False)
    if table.empty or table["track_id"].duplicated().any():
        raise ValueError("Таблица записей пуста или содержит повторяющиеся ID.")
    flags = table["metadata_valid"].astype(str).map({"True": True, "False": False})
    if flags.isna().any():
        raise ValueError("Некорректные значения metadata_valid.")
    table["metadata_valid"] = flags
    for row in table.itertuples():
        tid = f"{int(row.track_id):06d}"
        if row.audio_path != f"fma_small/{tid[:3]}/{tid}.mp3":
            raise ValueError(f"Неверный путь для записи {row.track_id}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"status": "running", "safe_for_training": False, "min_duration_seconds": min_duration}
    save_json(output_dir / "audio_summary.json", summary)
    tasks = [(record, str(data_dir), min_duration) for record in table.to_dict("records")]
    records = []
    def collect(results):
        for number, record in enumerate(results, 1):
            records.append(record)
            if number % 1000 == 0 or number == len(table):
                print(f"Проверено {number}/{len(table)}", flush=True)
    try:
        if workers == 1:
            collect(map(audit_track, tasks))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                collect(pool.map(audit_track, tasks, chunksize=4))
        audit = pd.DataFrame(records)
        for column in ("sample_rate", "channels", "frames"):
            audit[column] = audit[column].astype("Int64")
        audit, duplicates = remove_duplicates(audit)
        clean = audit.loc[audit["exclusion_reasons"].eq("")].copy()
        excluded = audit.loc[~audit["exclusion_reasons"].eq("")].copy()
        artist_splits = clean.groupby("artist_id")["split"].nunique()
        overlap = clean.loc[clean["artist_id"].isin(artist_splits[artist_splits > 1].index), ["artist_id", "track_id", "split"]]
        counts = pd.crosstab(clean["genre"], clean["split"]).reindex(index=GENRES, columns=("train", "validation", "test"), fill_value=0).fillna(0).astype(int)
        for name, frame in (("audio_audit", audit), ("tracks_clean", clean), ("tracks_excluded", excluded),
                            ("duplicates", duplicates), ("artist_overlap", overlap), ("class_counts", counts.reset_index())):
            save_csv(output_dir / f"{name}.csv", frame)
        save_json(output_dir / "labels.json", {genre: index for index, genre in enumerate(GENRES)})
        summary.update(status="completed", tracks_checked=len(table), kept=len(clean), excluded=len(excluded), excluded_by_combined_reason=excluded["exclusion_reasons"].value_counts().to_dict(), duplicate_groups=int(duplicates["duplicate_group"].nunique()), artist_overlap_count=int(overlap["artist_id"].nunique()), safe_for_training=bool(overlap.empty and (counts.to_numpy() > 0).all()))
    except BaseException:
        summary["status"] = "failed"
        raise
    finally:
        save_json(output_dir / "audio_summary.json", summary)
    print(f"Оставлено: {len(clean)}; исключено: {len(excluded)}. Результат: {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверить аудио и убрать ошибки и дубликаты.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-duration", type=float, default=29.0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if not math.isfinite(args.min_duration) or not 0 < args.min_duration <= 30 or args.workers < 1:
        parser.error("Нужны 0 < min-duration <= 30 и workers >= 1.")
    data_dir = args.data_dir.expanduser().resolve()
    try:
        summary = validate_dataset(data_dir, args.manifest or data_dir / "processed/tracks_metadata.csv", args.output_dir or data_dir / "processed", args.min_duration, args.workers)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Ошибка проверки аудио: {error}\n")
    if not summary["safe_for_training"]:
        parser.exit(2, "Проверьте пересечения исполнителей и наличие жанров в выборках.\n")


if __name__ == "__main__":
    main()

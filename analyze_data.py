import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import GENRES, SPLITS, save_csv, save_json

SPLIT_ORDER = ("train", "validation", "test")


def analyze_data(data_dir: Path, processed_dir: Path, output_dir: Path, make_plots: bool = True) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = []
    report = {"status": "running"}
    save_json(output_dir / "analysis_summary.json", report)
    def check(name, condition):
        status = "ok" if bool(condition) else "error"
        checks.append({"check": name, "status": status})
        print(f"[{'OK' if status == 'ok' else 'ОШИБКА'}] {name}", flush=True)
    try:
        if not (data_dir / "fma_small").is_dir():
            raise FileNotFoundError("Не найдена папка data/fma_small.")
        tables = {name: pd.read_csv(processed_dir / f"{name}.csv", keep_default_na=False) for name in ("tracks_metadata", "audio_audit", "tracks_clean", "tracks_excluded")}
        before, audit, clean, excluded = tables.values()
        labels = json.loads((processed_dir / "labels.json").read_text())
        summary = json.loads((processed_dir / "audio_summary.json").read_text())
        check("Подготовка и аудит завершены", summary.get("status") == "completed" and summary.get("safe_for_training") is True)
        check("ID записей не повторяются", all(not frame["track_id"].duplicated().any() for frame in tables.values()))
        source_ids, audit_ids, clean_ids, excluded_ids = (set(frame["track_id"]) for frame in tables.values())
        check("Все записи учтены без пересечения clean/excluded", source_ids == audit_ids == clean_ids | excluded_ids and not clean_ids & excluded_ids)
        check("Решения об исключении сохранены", set(audit.loc[audit["exclusion_reasons"].eq(""), "track_id"]) == clean_ids and set(audit.loc[audit["exclusion_reasons"].ne(""), "track_id"]) == excluded_ids)
        label_map = {genre: index for index, genre in enumerate(GENRES)}
        check("Метки жанров корректны", labels == label_map and clean["genre"].map(label_map).eq(clean["label"]).all())
        identity_ok = True
        for frame in (clean, excluded):
            joined = frame.merge(before, on="track_id", suffixes=("_result", "_source"), validate="one_to_one")
            for column in ("genre", "split", "label", "artist_id", "audio_path"):
                left, right = joined[f"{column}_result"], joined[f"{column}_source"]
                if column in ("label", "artist_id"):
                    left, right = pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")
                    identity_ok &= bool(((left == right) | (left.isna() & right.isna())).all())
                else:
                    identity_ok &= bool(left.eq(right).all())
        check("Метки, пути и исходные выборки сохранены", identity_ok and before["official_split"].map(SPLITS).fillna("").eq(before["split"]).all())
        check("Точные дубликаты исключены", clean["pcm_sha256"].ne("").all() and not clean["pcm_sha256"].duplicated().any())
        artists = pd.to_numeric(clean["artist_id"], errors="coerce")
        check("Исполнители не пересекаются между выборками", artists.notna().all() and artists.gt(0).all() and not clean.groupby("artist_id")["split"].nunique().gt(1).any())
        unavailable = []
        for row in clean.itertuples():
            tid = f"{int(row.track_id):06d}"
            expected_path = f"fma_small/{tid[:3]}/{tid}.mp3"
            path = data_dir / expected_path
            if row.audio_path != expected_path or not path.is_file() or path.stat().st_size == 0:
                unavailable.append(int(row.track_id))
        check("Аудиофайлы доступны", not unavailable)
        report["unavailable_audio_ids"] = unavailable
        columns = ["duration_seconds", "sample_rate", "channels", "frames"]
        clean[columns] = clean[columns].apply(pd.to_numeric, errors="coerce")
        valid_audio = bool(not clean.empty and np.isfinite(clean[columns].to_numpy(dtype=float)).all() and clean["duration_seconds"].ge(summary["min_duration_seconds"]).all() and clean[["sample_rate", "channels", "frames"]].gt(0).all().all())
        check("Параметры аудио корректны", valid_audio)
        if not valid_audio:
            raise ValueError("В таблице некорректные параметры аудио.")
        clean[["sample_rate", "channels", "frames"]] = clean[["sample_rate", "channels", "frames"]].astype(int)
        counts = pd.crosstab(clean["genre"], clean["split"]).reindex(index=GENRES, columns=SPLIT_ORDER, fill_value=0).fillna(0).astype(int)
        check("Все жанры представлены в train, validation и test", (counts.to_numpy() > 0).all() and clean["split"].isin(SPLIT_ORDER).all())
        genres = counts.reset_index()
        genres["before"] = genres["genre"].map(before["genre"].value_counts()).fillna(0).astype(int)
        genres["after"] = counts.sum(axis=1).to_numpy()
        genres["excluded"] = genres["before"] - genres["after"]
        reasons = excluded["exclusion_reasons"].value_counts().rename_axis("reason").reset_index(name="count")
        frequencies = clean["sample_rate"].value_counts().sort_index().rename_axis("sample_rate").reset_index(name="count")
        channels = clean["channels"].value_counts().sort_index().rename_axis("channels").reset_index(name="count")
        for name, frame in (("genre_distribution", genres), ("exclusion_reasons", reasons), ("sample_rates", frequencies), ("channels", channels)):
            save_csv(output_dir / f"{name}.csv", frame)
        durations = clean["duration_seconds"].describe().to_dict()
        report.update(before=len(before), after=len(clean), excluded=len(excluded), by_split=clean["split"].value_counts().to_dict(), duration_seconds=durations, sample_rates=frequencies.to_dict("records"), channels=channels.to_dict("records"))
        print(f"\nДо очистки: {len(before)}; после: {len(clean)}; исключено: {len(excluded)}")
        for title, frame in (("Жанры и выборки", genres), ("Причины исключений", reasons), ("Исходные частоты дискретизации", frequencies), ("Исходное число каналов", channels)):
            print(f"\n{title}:\n{frame.to_string(index=False)}")
        print(f"\nИсходная длительность: {durations['min']:.4f}–{durations['max']:.4f} с; средняя {durations['mean']:.4f} с")
        trim_path = processed_dir / "tracks_trimmed.csv"
        if trim_path.is_file():
            trimmed = pd.read_csv(trim_path)
            resampled = pd.read_csv(processed_dir / "tracks_resampled.csv")
            identity = ["genre", "label", "split", "artist_id"]
            preserved = len(trimmed) == len(clean) and trimmed["track_id"].is_unique and trimmed.set_index("track_id")[identity].sort_index().equals(clean.set_index("track_id")[identity].sort_index())
            check("Обрезка: ID, жанры и выборки сохранены", preserved)
            provenance = trimmed.merge(clean, on="track_id", suffixes=("_processed", "_original"), validate="one_to_one")
            source_ok = provenance["audio_path_processed"].eq(provenance["audio_path_original"]).all() and provenance["source_audio_path"].eq(provenance["audio_path_original"]).all() and provenance["source_sample_rate"].eq(provenance["sample_rate_original"]).all() and provenance["source_channels"].eq(provenance["channels_original"]).all() and provenance["source_frames"].eq(provenance["frames_original"]).all()
            check("Обработка: исходные пути, частоты, каналы и длины сохранены", source_ok)
            target_frames = int(resampled["frames"].min())
            valid_trim = trimmed["sample_rate"].eq(44100).all() and trimmed["channels"].eq(1).all() and trimmed["frames"].eq(target_frames).all() and trimmed["trim_start_frame"].eq(0).all() and trimmed["trim_end_frame"].eq(target_frames).all() and trimmed["cropped_frames"].eq(trimmed["frames_before_trim"] - target_frames).all()
            from trim_audio import load_trimmed_audio
            for row in trimmed.drop_duplicates("frames_before_trim").itertuples():
                load_trimmed_audio(data_dir / row.audio_path, target_frames)
            check("Обрезка при загрузке: единая длина, начало с нуля и частота 44 100 Гц", valid_trim)
            check("Обрезка: входные аудиофайлы доступны", all((data_dir / path).is_file() for path in trimmed["audio_path"]))
            report["trimmed_audio"] = dict(tracks=len(trimmed), sample_rate=44100, frames=target_frames, duration_seconds=target_frames / 44100, trimmed=int(trimmed["was_trimmed"].sum()), max_cropped_frames=int(trimmed["cropped_frames"].max()))
            print(f"\nПосле обрезки при загрузке: {target_frames} отсчётов; {target_frames / 44100:.8f} с; 44 100 Гц", flush=True)
        if make_plots:
            from plot_data import plot_overview
            plot_overview(clean, genres, reasons, Path(__file__).resolve().parent / "plots/overview.png")
    except (OSError, ValueError, KeyError, TypeError) as error:
        check(str(error), False)
    report["errors"] = sum(item["status"] == "error" for item in checks)
    report["status"] = "failed" if report["errors"] else "passed"
    save_csv(output_dir / "checks.csv", pd.DataFrame(checks))
    save_json(output_dir / "analysis_summary.json", report)
    print(f"\nОшибок: {report['errors']}. Отчёты: {output_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверить подготовленные данные и вывести статистику.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "reports/data_analysis")
    parser.add_argument("--no-plots", action="store_true", help="Пропустить графики; в пайплайне они строятся отдельным этапом.")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    try:
        report = analyze_data(data_dir, args.processed_dir or data_dir / "processed", args.output_dir, make_plots=not args.no_plots)
    except OSError as error:
        parser.exit(1, f"Ошибка записи отчёта: {error}\n")
    if report["errors"]:
        parser.exit(1, "Проверьте ошибки в отчёте checks.csv.\n")


if __name__ == "__main__":
    main()

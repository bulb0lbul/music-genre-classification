import argparse
import json
from pathlib import Path

import pandas as pd

GENRES = ("Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock")
SPLITS = {"training": "train", "validation": "validation", "test": "test"}
COLUMNS = {
    "genre": ("track", "genre_top"), "official_split": ("set", "split"),
    "artist_id": ("artist", "id"), "album_id": ("album", "id"),
    "title": ("track", "title"), "artist_name": ("artist", "name"),
    "license": ("track", "license"), "metadata_duration": ("track", "duration"),
}


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path: Path, table: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def positive_id(value) -> int | None:
    try:
        number = float(value)
        return int(number) if number > 0 and number.is_integer() else None
    except (TypeError, ValueError, OverflowError):
        return None


def prepare_metadata(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    tracks = pd.read_csv(data_dir / "fma_metadata/tracks.csv", header=[0, 1], index_col=0, low_memory=False)
    small = tracks.loc[tracks[("set", "subset")].eq("small")].sort_index()
    if small.empty or small.index.has_duplicates:
        raise ValueError("Поднабор small пуст или содержит повторяющиеся ID.")
    records = []
    for track_id, row in small.iterrows():
        track_id = positive_id(track_id)
        if track_id is None:
            raise ValueError("Некорректный ID записи в метаданных.")
        record = {name: row[column] for name, column in COLUMNS.items()}
        genre = record["genre"]
        record.update(track_id=track_id, subset="small", label=GENRES.index(genre) if genre in GENRES else -1)
        record["split"] = SPLITS.get(record["official_split"], "")
        record["artist_id"] = positive_id(record["artist_id"])
        record["album_id"] = positive_id(record["album_id"])
        tid = f"{track_id:06d}"
        record["audio_path"] = f"fma_small/{tid[:3]}/{tid}.mp3"
        path = data_dir / record["audio_path"]
        issues = []
        if record["label"] == -1:
            issues.append("missing_or_unknown_genre")
        if not record["split"]:
            issues.append("missing_or_unknown_split")
        if record["artist_id"] is None:
            issues.append("missing_or_invalid_artist")
        if not path.is_file():
            issues.append("missing_audio")
        elif path.stat().st_size == 0:
            issues.append("empty_file")
        record["metadata_valid"] = not issues
        record["metadata_issues"] = ";".join(issues)
        records.append(record)
    table = pd.DataFrame(records)
    table[["artist_id", "album_id"]] = table[["artist_id", "album_id"]].astype("Int64")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "tracks_metadata.csv", table)
    save_json(output_dir / "labels.json", {genre: index for index, genre in enumerate(GENRES)})
    save_json(output_dir / "metadata_summary.json", { "selected_tracks": len(table), "metadata_valid": int(table["metadata_valid"].sum()), "genre_counts": table["genre"].fillna("<missing>").value_counts().to_dict(), "split_counts": table["split"].value_counts().to_dict(), })
    print(f"Записей: {len(table)}; без ошибок метаданных: {table['metadata_valid'].sum()}")
    print(f"Результат: {output_dir}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовить метаданные FMA small.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true", help="Пропустить графики; в пайплайне они строятся отдельным этапом.")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    try:
        table = prepare_metadata(data_dir, args.output_dir or data_dir / "processed")
        if not args.no_plots:
            from plot_data import plot_metadata
            plot_metadata(table, Path(__file__).resolve().parent / "plots")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Ошибка подготовки данных: {error}\n")


if __name__ == "__main__":
    main()

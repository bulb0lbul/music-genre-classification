import argparse
from http.client import IncompleteRead
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile

ARCHIVES = {"metadata": "fma_metadata", "audio": "fma_small"}
BASE_URL = "https://os.unil.cloud.switch.ch/fma"


def dataset_ready(data_dir: Path, kind: str) -> bool:
    folder = data_dir / ARCHIVES[kind]
    if kind == "metadata":
        return all((folder / name).is_file() and (folder / name).stat().st_size > 0 for name in ("tracks.csv", "genres.csv"))
    files = list(folder.glob("*/*.mp3"))
    return len(files) == 8000 and all(path.is_file() and path.stat().st_size > 0 for path in files)


def download_archive(data_dir: Path, name: str) -> Path:
    archive_path = data_dir / f"{name}.zip"
    if archive_path.is_file():
        return archive_path

    partial_path = archive_path.with_suffix(".zip.part")
    print(f"Скачиваю {name}…", flush=True)
    downloaded = 0
    next_message = 64 * 1024**2
    with urllib.request.urlopen(f"{BASE_URL}/{name}.zip", timeout=60) as response:
        with partial_path.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_message:
                    print(f"Загружено {downloaded // 1024**2} MiB", flush=True)
                    next_message += 64 * 1024**2
    partial_path.replace(archive_path)
    return archive_path


def extract_archive(archive_path: Path, data_dir: Path, name: str, kind: str) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        if kind == "metadata":
            members = [archive.getinfo(f"{name}/{file}") for file in ("tracks.csv", "genres.csv")]
        else:
            members = [item for item in archive.infolist() if item.filename.startswith(f"{name}/") and item.filename.endswith(".mp3")]
        if not members:
            raise ValueError(f"В архиве {archive_path.name} нет нужных файлов.")
        for item in members:
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                raise ValueError(f"Недопустимый путь в архиве: {item.filename}")
        print(f"Распаковываю {name}…", flush=True)
        with tempfile.TemporaryDirectory(dir=data_dir) as temporary:
            temporary_dir = Path(temporary)
            for item in members:
                target = temporary_dir / item.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
            (temporary_dir / name).rename(data_dir / name)


def download_dataset(data_dir: Path, only: str = "all") -> None:
    data_dir = data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    kinds = ("metadata", "audio") if only == "all" else (only,)
    for kind in kinds:
        name = ARCHIVES[kind]
        folder = data_dir / name
        if dataset_ready(data_dir, kind):
            print(f"Папка уже есть, пропускаю: {folder}", flush=True)
            continue
        if folder.exists():
            raise ValueError(f"Набор в {folder} неполон: для метаданных нужны непустые tracks.csv и genres.csv, для FMA small — 8 000 непустых MP3. Проверьте содержимое папки перед повторным скачиванием.")
        archive_path = download_archive(data_dir, name)
        extract_archive(archive_path, data_dir, name, kind)
        print(f"Готово: {folder}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать FMA small и метаданные.")
    parser.add_argument("--only", choices=("all", "metadata", "audio"), default="all")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    try:
        download_dataset(args.data_dir, args.only)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, IncompleteRead) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()

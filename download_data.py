"""Скачать FMA small и метаданные в локальную папку data/."""

import argparse
from dataclasses import dataclass
import hashlib
from http.client import IncompleteRead
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile


@dataclass(frozen=True)
class Archive:
    name: str
    sha1: str
    minimum_free_gib: int


# Источник URL и SHA-1: https://github.com/mdeff/fma#data
ARCHIVES = {
    "metadata": Archive("fma_metadata", "f0df49ffe5f2a6008d7dc83c6915b31835dfe733", 1),
    "audio": Archive("fma_small", "ade154f733639d52e35e32f5593efe5be76c6d70", 16),
}
BASE_URL = "https://os.unil.cloud.switch.ch/fma"
CHUNK_SIZE = 1024 * 1024


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_data(folder: Path, kind: str) -> bool:
    """Проверить наличие файлов; это не проверка декодирования аудио."""
    if not folder.exists():
        return False
    if kind == "metadata":
        paths = [folder / name for name in ("tracks.csv", "genres.csv")]
    else:
        paths = list(folder.rglob("*.mp3"))
        if len(paths) != 8000:
            raise ValueError(
                f"В {folder} найдено {len(paths)} MP3 вместо 8000. "
                "Папка не изменена. Переместите её перед повторной загрузкой."
            )
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        raise ValueError(
            f"В {folder} отсутствуют нужные файлы или есть пустые файлы. "
            "Папка не изменена. Переместите её перед повторной загрузкой."
        )
    return True


def download_archive(data_dir: Path, archive: Archive) -> Path:
    archive_path = data_dir / f"{archive.name}.zip"
    if archive_path.exists():
        print(f"Проверяю сохранённый архив {archive_path.name}…", flush=True)
        if sha1_file(archive_path) != archive.sha1:
            raise ValueError(
                f"SHA-1 не совпал: {archive_path}. "
                "Удалите или переместите повреждённый архив и повторите запуск."
            )
        return archive_path

    if shutil.disk_usage(data_dir).free < archive.minimum_free_gib * 1024**3:
        raise OSError(
            f"Для {archive.name} нужно хотя бы {archive.minimum_free_gib} GiB "
            "свободного места под архив и распаковку."
        )
    url = f"{BASE_URL}/{archive.name}.zip"
    partial_path = archive_path.with_suffix(".zip.part")
    print(f"Скачиваю {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "music-genre-classification"})
    digest = hashlib.sha1()
    downloaded = 0
    next_message = 64 * CHUNK_SIZE
    with urllib.request.urlopen(request, timeout=60) as response:
        with partial_path.open("wb") as target:
            while chunk := response.read(CHUNK_SIZE):
                target.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if downloaded >= next_message:
                    print(f"Загружено {downloaded / CHUNK_SIZE:.0f} MiB", flush=True)
                    next_message += 64 * CHUNK_SIZE
    if digest.hexdigest() != archive.sha1:
        raise ValueError(
            f"SHA-1 не совпал для {archive.name}. Файл оставлен как {partial_path.name}. "
            "При повторном запуске загрузка начнётся заново."
        )
    partial_path.replace(archive_path)
    print("SHA-1 совпадает с опубликованным авторами.", flush=True)
    return archive_path


def extract_archive(archive_path: Path, data_dir: Path, archive: Archive, kind: str) -> Path:
    folder = data_dir / archive.name
    if folder.exists():
        raise FileExistsError(f"Папка уже существует, не перезаписываю: {folder}")
    with zipfile.ZipFile(archive_path) as source:
        if kind == "metadata":
            members = [source.getinfo(f"{archive.name}/{name}") for name in ("tracks.csv", "genres.csv")]
        else:
            members = [
                member for member in source.infolist()
                if member.filename.startswith(f"{archive.name}/")
                and member.filename.lower().endswith(".mp3") and not member.is_dir()
            ]
            if len(members) != 8000:
                raise ValueError(f"В архиве найдено {len(members)} MP3 вместо 8000.")
        names = set()
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError(f"Некорректный путь внутри архива: {member.filename}")
            if member.filename in names or member.file_size == 0:
                raise ValueError(f"Пустая или повторная запись в архиве: {member.filename}")
            names.add(member.filename)
        required = sum(member.file_size for member in members)
        if shutil.disk_usage(data_dir).free < required:
            raise OSError("Недостаточно места для распаковки.")
        print(f"Распаковываю {archive.name}: {len(members)} файлов…", flush=True)
        # Готовая папка появляется только после успешной распаковки всех файлов.
        with tempfile.TemporaryDirectory(prefix=f".{archive.name}-", dir=data_dir) as temporary:
            temporary_dir = Path(temporary)
            for member in members:
                target = temporary_dir / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output)
            (temporary_dir / archive.name).rename(folder)
    return folder


def prepare_data(data_dir: Path, only: str = "all") -> None:
    data_dir = data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    kinds = ("metadata", "audio") if only == "all" else (only,)
    for kind in kinds:
        archive = ARCHIVES[kind]
        folder = data_dir / archive.name
        if existing_data(folder, kind):
            print(
                f"Файлы уже есть: {folder}. Скачивание пропущено; "
                "проверено только наличие и ненулевой размер файлов.", flush=True,
            )
            continue
        archive_path = download_archive(data_dir, archive)
        folder = extract_archive(archive_path, data_dir, archive, kind)
        print(f"Готово: {folder}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=("all", "metadata", "audio"), default="all",
        help="Что скачать: all (по умолчанию), metadata или audio (FMA small).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data",
        help="Папка данных; по умолчанию data/ рядом со скриптом.",
    )
    args = parser.parse_args()
    try:
        prepare_data(args.data_dir, args.only)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, IncompleteRead) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()

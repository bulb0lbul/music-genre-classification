import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_DIR = Path(__file__).resolve().parent


def run_script(name: str, data_dir: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(PROJECT_DIR / name), "--data-dir", str(data_dir), *args], cwd=PROJECT_DIR, check=True)


def run_pipeline(data_dir: Path) -> None:
    metadata_dir = data_dir / "fma_metadata"
    audio_dir = data_dir / "fma_small"
    metadata_ready = all((metadata_dir / name).is_file() for name in ("tracks.csv", "genres.csv"))
    audio_ready = audio_dir.is_dir() and next(audio_dir.glob("*/*.mp3"), None) is not None
    for kind, folder, ready in (("metadata", metadata_dir, metadata_ready), ("audio", audio_dir, audio_ready)):
        if ready:
            print(f"Данные уже есть, скачивание пропущено: {folder}", flush=True)
        elif folder.exists():
            raise ValueError(f"Папка {folder} существует, но нужные файлы не найдены. Переместите её и повторите запуск.")
        else:
            print(f"Данные не найдены, запускаю загрузку: {kind}", flush=True)
            run_script("download_data.py", data_dir, "--only", kind)
    stages = (("Подготовка метаданных", "prepare_data.py"), ("Проверка аудио", "validate_audio.py"), ("Анализ результатов", "analyze_data.py"))
    for number, (title, script) in enumerate(stages, 1):
        print(f"\n[{number}/{len(stages)}] {title}", flush=True)
        run_script(script, data_dir)
    print("\nОбработка завершена.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать недостающие данные и запустить всю обработку.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    args = parser.parse_args()
    try:
        run_pipeline(args.data_dir.expanduser().resolve())
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode if error.returncode > 0 else 1, f"Обработка остановлена: ошибка в {Path(error.cmd[1]).name}.\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nОбработка прервана.\n")


if __name__ == "__main__":
    main()

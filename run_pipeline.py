import argparse
from pathlib import Path
import subprocess
import sys

from download_data import ARCHIVES, dataset_ready

PROJECT_DIR = Path(__file__).resolve().parent


def run_script(name: str, data_dir: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(PROJECT_DIR / name), "--data-dir", str(data_dir), *args], cwd=PROJECT_DIR, check=True)


def run_pipeline(data_dir: Path) -> None:
    for kind, name in ARCHIVES.items():
        if dataset_ready(data_dir, kind):
            print(f"Данные уже есть, скачивание пропущено: {data_dir / name}", flush=True)
        else:
            print(f"Проверка и загрузка набора: {kind}", flush=True)
            run_script("download_data.py", data_dir, "--only", kind)
    stages = (("Подготовка метаданных", "prepare_data.py", ("--no-plots",)), ("Проверка аудио", "validate_audio.py", ("--no-plots",)), ("Подготовка сведения в моно при загрузке", "process_audio.py", ("--all", "--no-plots")), ("Подготовка ресемплинга до 44 100 Гц при загрузке", "resample_audio.py", ("--no-plots",)), ("Подготовка фрагментов минимальной длины", "trim_audio.py", ("--no-plots",)), ("Проверка примеров и подготовка прослушивания", "check_audio.py", ("--no-plots",)), ("Анализ результатов", "analyze_data.py", ("--no-plots",)), ("Построение графиков", "plot_data.py", ("--kind", "all")))
    for number, (title, script, extra_args) in enumerate(stages, 1):
        print(f"\n[{number}/{len(stages)}] {title}", flush=True)
        run_script(script, data_dir, *extra_args)
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

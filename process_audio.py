import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf


def load_audio(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Аудиофайл не найден: {path}")
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if len(audio) == 0:
        raise ValueError("Аудиофайл не содержит отсчётов.")
    return audio, sample_rate


def to_mono(audio):
    return audio.mean(axis=1)


def load_mono_audio(path: Path):
    audio, sample_rate = load_audio(path)
    return to_mono(audio), sample_rate


def prepare_mono_dataset(data_dir: Path, make_plots: bool = True) -> None:
    tracks = pd.read_csv(data_dir / "processed/tracks_clean.csv")
    records = []
    for row in tracks.itertuples():
        if not (data_dir / row.audio_path).is_file():
            raise FileNotFoundError(f"Аудиофайл не найден: {row.audio_path}")
        records.append(dict(track_id=row.track_id, genre=row.genre, label=row.label, split=row.split, artist_id=row.artist_id, source_audio_path=row.audio_path, audio_path=row.audio_path, source_channels=row.channels, sample_rate=row.sample_rate, channels=1, frames=row.frames, duration_seconds=row.duration_seconds))
    path = data_dir / "processed/tracks_mono.csv"
    processed = pd.DataFrame(records)
    processed.to_csv(path, index=False)
    if make_plots:
        from plot_data import plot_mono_dataset, plot_waveforms, create_example_plots
        plot_mono_dataset(tracks, processed, data_dir, Path(__file__).resolve().parent / "plots/mono")
        plot_waveforms(processed, data_dir, Path(__file__).resolve().parent / "plots/random_waveforms.png")
        create_example_plots(data_dir, Path(__file__).resolve().parent / "plots")
    print(f"Подготовлено для сведения в моно при загрузке: {len(processed)} записей; WAV-копии не создаются.", flush=True)
    print(f"Таблица: {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Перевести одну запись или очищенный набор в моно.")
    parser.add_argument("audio_path", type=Path, nargs="?")
    parser.add_argument("--all", action="store_true", help="Подготовить таблицу для сведения очищенного набора в моно при загрузке.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--no-plots", action="store_true", help="Пропустить графики; в пайплайне они строятся отдельным этапом.")
    args = parser.parse_args()
    if args.all and args.audio_path:
        parser.error("Укажите путь к одной записи или --all.")
    if not args.all and not args.audio_path:
        parser.error("Укажите путь к записи или --all.")
    try:
        if args.all:
            prepare_mono_dataset(args.data_dir.expanduser().resolve(), make_plots=not args.no_plots)
        else:
            path = args.audio_path.expanduser().resolve()
            audio, sample_rate = load_audio(path)
            mono = to_mono(audio)
            print(f"Файл: {path}")
            print(f"Частота дискретизации: {sample_rate} Гц")
            print(f"Каналов до обработки: {audio.shape[1]}; после: 1")
            print(f"Длительность: {len(mono) / sample_rate:.4f} с")
            print(f"Форма массива: {mono.shape}")
            if not args.no_plots:
                from plot_data import plot_mono_example
                figure = Path(__file__).resolve().parent / "plots/mono" / f"track_{path.stem}.png"
                plot_mono_example(audio, mono, sample_rate, figure, f"{path.stem} · {audio.shape[1]} → 1 канал · {sample_rate} Гц")
                print(f"График: {figure}")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ошибка обработки аудио: {error}\n")


if __name__ == "__main__":
    main()

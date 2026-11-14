import argparse
import base64
import html
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from process_audio import load_audio, to_mono
from resample_audio import TARGET_SAMPLE_RATE, load_resampled_audio
from trim_audio import load_trimmed_audio


def select_examples(tracks: pd.DataFrame) -> pd.DataFrame:
    formats = tracks.drop_duplicates(["source_sample_rate", "source_channels"])
    lengths = tracks.loc[[tracks["frames_before_trim"].idxmin(), tracks["frames_before_trim"].idxmax()]]
    return pd.concat([formats, lengths]).drop_duplicates("track_id").sort_values(["source_sample_rate", "source_channels", "track_id"])


def envelope_correlation(before, source_rate: int, after, target_rate: int) -> float:
    source_size, target_size = source_rate // 10, target_rate // 10
    count = min(len(before) // source_size, len(after) // target_size)
    source = before[:count * source_size].reshape(count, source_size).astype(np.float64)
    target = after[:count * target_size].reshape(count, target_size).astype(np.float64)
    source_rms = np.sqrt(np.mean(source * source, axis=1))
    target_rms = np.sqrt(np.mean(target * target, axis=1))
    if source_rms.std() == 0 or target_rms.std() == 0:
        return float("nan")
    return float(np.corrcoef(source_rms, target_rms)[0, 1])


def audio_player(audio, sample_rate: int) -> str:
    stream = BytesIO()
    sf.write(stream, audio, sample_rate, format="WAV", subtype="FLOAT")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f'<audio controls preload="none" src="data:audio/wav;base64,{encoded}"></audio>'


def check_examples(data_dir: Path, output_dir: Path, make_plots: bool = True) -> pd.DataFrame:
    tracks = pd.read_csv(data_dir / "processed/tracks_trimmed.csv")
    if tracks.empty:
        raise ValueError("Нет записей для проверки. Сначала выполните trim_audio.py.")
    output_dir.mkdir(parents=True, exist_ok=True)
    checks, cards = [], []
    for row in select_examples(tracks).itertuples():
        source, source_rate = load_audio(data_dir / row.audio_path)
        mono = to_mono(source)
        resampled, target_rate = load_resampled_audio(data_dir / row.audio_path)
        processed, processed_rate = load_trimmed_audio(data_dir / row.audio_path, int(row.frames))
        expected_frames = (len(source) * TARGET_SAMPLE_RATE + source_rate - 1) // source_rate
        duration_error = len(resampled) / target_rate - len(source) / source_rate
        source_ok = source_rate == row.source_sample_rate and source.shape == (row.source_frames, row.source_channels)
        format_ok = processed_rate == TARGET_SAMPLE_RATE and processed.shape == (row.frames,) and processed.dtype == np.float32 and np.isfinite(processed).all() and np.any(processed != 0)
        timing_ok = target_rate == TARGET_SAMPLE_RATE and len(resampled) == expected_frames and abs(duration_error) < 1 / TARGET_SAMPLE_RATE
        prefix_ok = np.array_equal(processed, resampled[:int(row.frames)])
        if source_rate == TARGET_SAMPLE_RATE:
            prefix_ok = prefix_ok and np.array_equal(processed, mono[:int(row.frames)])
        correlation = envelope_correlation(mono, source_rate, processed, processed_rate)
        passed = bool(source_ok and format_ok and timing_ok and prefix_ok)
        start = min(10.0, max(0.0, len(processed) / processed_rate - 5.0))
        end = min(start + 5.0, len(processed) / processed_rate)
        original_clip = source[round(start * source_rate):round(end * source_rate)]
        processed_clip = processed[round(start * processed_rate):round(end * processed_rate)]
        gain = 1 / max(1.0, float(np.max(np.abs(original_clip))), float(np.max(np.abs(processed_clip))))
        checks.append(dict(track_id=row.track_id, genre=row.genre, source_sample_rate=source_rate, source_channels=source.shape[1], source_shape=str(source.shape), processed_shape=str(processed.shape), source_duration_seconds=len(source) / source_rate, resampled_duration_seconds=len(resampled) / target_rate, processed_duration_seconds=len(processed) / processed_rate, processed_sample_rate=processed_rate, cropped_frames=len(resampled) - len(processed), source_matches_manifest=bool(source_ok), processed_format_ok=bool(format_ok), duration_preserved_before_trim=bool(timing_ok), beginning_preserved=bool(prefix_ok), envelope_correlation=correlation, passed=passed, listening_start_seconds=start, listening_end_seconds=end, listening_gain=gain, listening_status="pending"))
        status = "OK" if passed else "ОШИБКА"
        print(f"[{status}] {row.track_id:06d} · {row.genre}: {source.shape} / {source_rate} Гц → {processed.shape} / {processed_rate} Гц", flush=True)
        print(f"     Длительность: {len(source) / source_rate:.8f} → {len(resampled) / target_rate:.8f} → {len(processed) / processed_rate:.8f} с; корреляция RMS: {correlation:.6f}", flush=True)
        plot_path = Path(__file__).resolve().parent / "plots/audio_check" / f"track_{row.track_id:06d}.png"
        plot_link = html.escape(os.path.relpath(plot_path, output_dir), quote=True)
        original_player = audio_player(original_clip * gain, source_rate)
        processed_player = audio_player(processed_clip * gain, processed_rate)
        cards.append(f'<section><h2>{row.track_id:06d} · {html.escape(row.genre)} · {status}</h2><p>Исходный массив: {source.shape}, {source_rate} Гц. После обработки: {processed.shape}, {processed_rate} Гц.</p><p>Одинаковый участок {start:g}–{end:g} с. Общий коэффициент громкости пары: {gain:.6f}.</p><div><label>Исходная запись · {source.shape[1]} кан.</label>{original_player}</div><div><label>Моно после ресемплинга и обрезки</label>{processed_player}</div><p>Корреляция RMS-огибающих: {correlation:.6f}. <a href="{plot_link}">График сравнения</a></p></section>')
        if make_plots:
            from plot_data import plot_audio_check
            plot_audio_check(row, source, source_rate, processed, processed_rate, plot_path)
    report = pd.DataFrame(checks)
    report.to_csv(output_dir / "examples.csv", index=False)
    body = "\n".join(cards)
    document = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка обработки аудио</title><style>body {{font:16px system-ui,sans-serif;max-width:960px;margin:36px auto;padding:0 20px;line-height:1.6;color:#202830}} section {{border-top:1px solid #ccc;margin-top:28px;padding-top:12px}} label {{display:block;font-weight:600}} audio {{width:min(100%,560px);margin:8px 0 16px}} a {{color:#21688b}}</style><h1>Проверка обработки аудио</h1><p>Для каждой записи сравниваются один и тот же пятисекундный участок до и после обработки. Исходное стерео сохранено. Обоим фрагментам пары применяется одинаковый коэффициент громкости, чтобы избежать перегрузки при прослушивании; рабочие массивы не меняются.</p><p>Запускайте плееры по очереди на скорости 1×. Сравните темп и высоту звука, отсутствие щелчков и искажений. Пространственное звучание при сведении стерео в моно изменится. Автоматическая проверка длительности и графики дополняют проверку на слух; прослушивание ещё не отмечено выполненным.</p>{body}</html>'''
    (output_dir / "index.html").write_text(document, encoding="utf-8")
    print(f"\nАвтоматических ошибок: {(~report['passed']).sum()}. Для прослушивания откройте: {output_dir / 'index.html'}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверить параметры обработки примеров и подготовить сравнение для прослушивания.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "reports/audio_check")
    parser.add_argument("--no-plots", action="store_true", help="Пропустить графики; в пайплайне они строятся отдельным этапом.")
    args = parser.parse_args()
    try:
        report = check_examples(args.data_dir.expanduser().resolve(), args.output_dir.expanduser().resolve(), make_plots=not args.no_plots)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ошибка проверки аудио: {error}\n")
    if not report["passed"].all():
        parser.exit(1, "Проверьте результаты в examples.csv.\n")


if __name__ == "__main__":
    main()

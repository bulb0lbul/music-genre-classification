import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_audio import load_audio, load_mono_audio, rms_envelope, to_mono
from resample_audio import TARGET_SAMPLE_RATE, load_resampled_audio, resample_signal
from trim_audio import load_trimmed_audio, select_check_examples, trim_signal


def plot_metadata(tracks: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    genres = tracks["genre"].value_counts().sort_index()
    bars = axes[0].barh(genres.index, genres.values, color="#337c9e")
    axes[0].bar_label(bars, padding=4)
    axes[0].set(title="Исходные записи по жанрам", xlabel="Количество записей", xlim=(0, genres.max() * 1.15))
    splits = tracks["split"].value_counts().reindex(["train", "validation", "test"], fill_value=0)
    bars = axes[1].bar(splits.index, splits.values, color=["#337c9e", "#d99547", "#537b63"])
    axes[1].bar_label(bars, padding=4)
    axes[1].set(title="Официальные выборки до очистки", ylabel="Количество записей", ylim=(0, splits.max() * 1.15))
    fig.savefig(output_dir / "metadata_overview.png", dpi=150)
    plt.close(fig)


def plot_validation(audit: pd.DataFrame, clean: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded = audit.loc[audit["exclusion_reasons"].ne("")].copy()
    def reason_name(value):
        if "decode_error" in value.split(";"):
            return "Не декодируются"
        names = {"decoder_reported_error": "Ошибки MPEG-кадров", "duplicate_same_split": "Повторные записи", "duplicate_label_conflict": "Конфликтующие жанры", "duplicate_across_splits": "Дубликаты между выборками"}
        return names.get(value, value)
    reasons = excluded["exclusion_reasons"].map(reason_name).value_counts().sort_values()
    lost = excluded["genre"].value_counts().reindex(sorted(audit["genre"].dropna().unique()), fill_value=0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    bars = axes[0].bar(["Оставлены", "Исключены"], [len(clean), len(excluded)], color=["#537b63", "#ba7866"])
    axes[0].bar_label(bars, padding=4)
    axes[0].set(title="Результат проверки аудио", ylabel="Количество записей", ylim=(0, len(audit) * 1.15))
    bars = axes[1].barh(reasons.index, reasons.values, color="#ba7866")
    axes[1].bar_label(bars, padding=4)
    axes[1].set(title="Причины исключения", xlabel="Количество записей", xlim=(0, max(1, reasons.max() if not reasons.empty else 1) * 1.2))
    bars = axes[2].barh(lost.index, lost.values, color="#d99547")
    axes[2].bar_label(bars, padding=4)
    axes[2].set(title="Исключения по жанрам", xlabel="Количество записей", xlim=(0, max(1, lost.max()) * 1.2))
    fig.savefig(output_dir / "audio_validation.png", dpi=150)
    plt.close(fig)


def plot_mono_example(audio, mono, sample_rate: int, path: Path, title: str) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    times, original_rms = rms_envelope(audio, sample_rate)
    _, mono_rms = rms_envelope(mono, sample_rate)
    size = max(1, sample_rate // 10)
    start = min(int(np.argmax(original_rms)) * size, max(0, len(audio) - int(0.04 * sample_rate)))
    end = min(len(audio), start + max(1, int(0.04 * sample_rate)))
    segment_time = np.arange(start, end) / sample_rate
    limit = max(float(np.max(np.abs(audio[start:end]))), float(np.max(np.abs(mono[start:end]))), 1e-6) * 1.1
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), layout="constrained")
    for channel in range(audio.shape[1]):
        label = "Исходное моно" if audio.shape[1] == 1 else ("Левый канал" if channel == 0 else "Правый канал")
        axes[0].plot(segment_time, audio[start:end, channel], label=label, linewidth=1, alpha=0.85)
    axes[0].legend(loc="upper right")
    axes[0].set(title="Исходный сигнал — фрагмент 40 мс", ylabel="Амплитуда", ylim=(-limit, limit))
    axes[1].plot(segment_time, mono[start:end], color="#537b63", linewidth=1)
    axes[1].set(title="После усреднения каналов — тот же фрагмент", xlabel="Время записи, с", ylabel="Амплитуда", ylim=(-limit, limit))
    axes[0].set_xlim(segment_time[0], segment_time[-1])
    axes[1].set_xlim(segment_time[0], segment_time[-1])
    axes[2].plot(times, 20 * np.log10(np.maximum(original_rms, 1e-12)), label="До — энергия усреднена по каналам", color="#337c9e")
    axes[2].plot(times, 20 * np.log10(np.maximum(mono_rms, 1e-12)), label="После — моно", color="#537b63", alpha=0.85)
    axes[2].set(title="RMS-уровень всей записи — окна 100 мс", xlabel="Время записи, с", ylabel="Уровень, дБFS")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.2)
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    before_power = float(np.mean(np.square(audio, dtype=np.float64)))
    after_power = float(np.mean(np.square(mono, dtype=np.float64)))
    return 10 * np.log10(max(after_power, 1e-24) / max(before_power, 1e-24))


def plot_mono_dataset(original: pd.DataFrame, processed: pd.DataFrame, data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = original.drop_duplicates(["channels", "sample_rate"]).head(6)
    rows = []
    for row in examples.itertuples():
        audio, sr = load_audio(data_dir / row.audio_path)
        mono = to_mono(audio)
        path = output_dir / f"track_{row.track_id:06d}.png"
        change = plot_mono_example(audio, mono, sr, path, f"{row.track_id:06d} · {row.genre} · {audio.shape[1]} → 1 канал · {sr} Гц")
        rows.append(dict(track_id=row.track_id, genre=row.genre, source_channels=audio.shape[1], sample_rate=sr, rms_change_db=change))
    selected = pd.DataFrame(rows)
    report_dir = Path(__file__).resolve().parent / "reports/mono"
    report_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(report_dir / "mono_examples.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
    positions = np.arange(2)
    before = [int(original["channels"].eq(1).sum()), int(original["channels"].eq(2).sum())]
    after = [int(processed["channels"].eq(1).sum()), int(processed["channels"].eq(2).sum())]
    for shift, values, label, color in ((-0.2, before, "До", "#337c9e"), (0.2, after, "После", "#537b63")):
        bars = axes[0].bar(positions + shift, values, width=0.4, label=label, color=color)
        axes[0].bar_label(bars, padding=3)
    axes[0].set(xticks=positions, xticklabels=["Моно", "Стерео"], title="Число каналов — весь очищенный набор", ylabel="Количество записей", ylim=(0, len(processed) * 1.15))
    axes[0].legend()
    labels = [f"{row.track_id:06d} · {row.source_channels} кан. · {row.sample_rate} Гц" for row in selected.itertuples()]
    bars = axes[1].barh(labels, selected["rms_change_db"], color="#d99547")
    axes[1].bar_label(bars, fmt="%.2f", padding=4)
    axes[1].axvline(0, color="black", linewidth=0.7)
    axes[1].set(title="Изменение RMS-уровня — выбранные примеры", xlabel="После минус до, дБ")
    lower = min(-0.05, float(selected["rms_change_db"].min()))
    upper = max(0.05, float(selected["rms_change_db"].max()))
    padding = (upper - lower) * 0.15
    axes[1].set_xlim(lower - padding, upper + padding)
    fig.savefig(output_dir / "mono_overview.png", dpi=150)
    plt.close(fig)


def plot_waveforms(tracks: pd.DataFrame, data_dir: Path, path: Path, seed=None) -> None:
    selected = tracks.sample(n=min(3, len(tracks)), random_state=seed)
    signals = [load_mono_audio(data_dir / row.audio_path) for row in selected.itertuples()]
    limit = max(1.0, max(float(np.max(np.abs(audio))) for audio, _ in signals) * 1.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(selected), 1, figsize=(14, 9), sharex=True, sharey=True, squeeze=False, layout="constrained")
    for axis, row, (audio, sample_rate) in zip(axes[:, 0], selected.itertuples(), signals):
        axis.plot(np.arange(len(audio)) / sample_rate, audio, color="#337c9e", linewidth=0.4)
        axis.set(title=f"{row.track_id:06d} · {row.genre} · {sample_rate} Гц · {len(audio) / sample_rate:.2f} с", ylabel="Амплитуда", ylim=(-limit, limit))
        axis.axhline(0, color="black", linewidth=0.4)
        axis.grid(alpha=0.2)
        axis.margins(x=0)
    axes[-1, 0].set_xlabel("Время, с")
    fig.suptitle("Звуковая волна трёх случайных записей — моно", fontsize=15)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"График звуковых волн: {path}", flush=True)
    print("Выбранные записи: " + ", ".join(f"{tid:06d}" for tid in selected["track_id"]), flush=True)


def select_genre_examples(tracks: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    candidates, features = [], []
    for _, group in tracks.loc[tracks["split"].eq("train")].groupby("genre"):
        for row in group.sample(n=min(40, len(group)), random_state=42).itertuples():
            audio, sample_rate = load_mono_audio(data_dir / row.audio_path)
            _, levels = rms_envelope(audio, sample_rate)
            relative = levels / max(float(levels.max()), 1e-12)
            features.append(np.r_[np.quantile(relative, [0.1, 0.5, 0.9]), relative.std(), np.mean(relative < 0.15), [part.mean() for part in np.array_split(relative, 6)]])
            candidates.append(row.track_id)
    values = np.array(features)
    values = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-6)
    rows = tracks.set_index("track_id").loc[candidates].reset_index()
    selected = [int(np.argmax(np.linalg.norm(values, axis=1)))]
    while rows.iloc[selected]["genre"].nunique() < rows["genre"].nunique():
        available = ~rows["genre"].isin(rows.iloc[selected]["genre"])
        distances = np.linalg.norm(values[:, None] - values[selected][None, :], axis=2).min(axis=1)
        selected.append(int(np.argmax(np.where(available, distances, -1))))
    return rows.iloc[selected].sort_values("genre")


def plot_genre_examples(tracks: pd.DataFrame, data_dir: Path, path: Path) -> None:
    selected = select_genre_examples(tracks, data_dir)
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True, sharey=True, layout="constrained")
    for axis, row, color in zip(axes.flat, selected.itertuples(), plt.get_cmap("tab10").colors):
        audio, sample_rate = load_mono_audio(data_dir / row.audio_path)
        normalized = audio / max(float(np.max(np.abs(audio))), 1e-12)
        axis.plot(np.arange(len(audio)) / sample_rate, normalized, color=color, linewidth=0.35)
        axis.set(title=f"{row.genre} · {row.track_id:06d}", ylim=(-1.05, 1.05), ylabel="Амплитуда / пик")
        axis.margins(x=0)
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Время, с")
    fig.suptitle("Разные жанры — примеры с различающейся формой сигнала", fontsize=15)
    fig.supxlabel("Масштаб каждой волны приведён к её пику только на рисунке. Примеры специально подобраны и не характеризуют жанр целиком.", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Примеры жанров: " + ", ".join(f"{row.genre}: {row.track_id:06d}" for row in selected.itertuples()), flush=True)


def plot_sample_rates(original: pd.DataFrame, tracks: pd.DataFrame, data_dir: Path, path: Path) -> None:
    import librosa

    row = tracks.loc[tracks["sample_rate"].eq(tracks["sample_rate"].max())].iloc[0]
    audio, source_rate = load_mono_audio(data_dir / row.audio_path)
    _, levels = rms_envelope(audio, source_rate)
    start = int(np.argmax(levels)) * (source_rate // 10)
    fragment = audio[start:start + source_rate // 5]
    fig = plt.figure(figsize=(14, 8), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 2])
    counts_axis = fig.add_subplot(grid[:, 0])
    counts = original["sample_rate"].value_counts().sort_index()
    bars = counts_axis.bar(counts.index.astype(str), counts.values, color="#337c9e")
    counts_axis.bar_label(bars, padding=5)
    counts_axis.set(title="Исходные частоты очищенного набора", xlabel="Частота дискретизации, Гц", ylabel="Количество записей", ylim=(0, counts.max() * 1.15))
    reference_time = np.arange(len(fragment)) / source_rate
    reference_mask = (reference_time >= 0.05) & (reference_time < 0.052)
    limit = float(np.max(np.abs(fragment))) * 1.15
    for index, rate in enumerate(counts.index):
        axis = fig.add_subplot(grid[index, 1])
        sampled = librosa.resample(fragment, orig_sr=source_rate, target_sr=int(rate)) if rate != source_rate else fragment
        time = np.arange(len(sampled)) / rate
        mask = (time >= 0.05) & (time < 0.052)
        axis.plot((reference_time[reference_mask] - 0.05) * 1000, fragment[reference_mask], color="#b8b8b8", linewidth=1, label=f"Исходные отсчёты {source_rate} Гц")
        axis.plot((time[mask] - 0.05) * 1000, sampled[mask], "o-", color="#337c9e", linewidth=1, markersize=3)
        axis.set(title=f"{rate:,} Гц · {int(mask.sum())} отсчётов на показанном интервале".replace(",", " "), ylabel="Амплитуда", xlim=(0, 2), ylim=(-limit, limit))
        axis.grid(alpha=0.2)
        if index == 0:
            axis.legend(fontsize=8, loc="upper right")
        if index == len(counts) - 1:
            axis.set_xlabel("Время от начала фрагмента, мс")
    fig.suptitle("Частоты дискретизации — состав набора и плотность отсчётов", fontsize=15)
    fig.supxlabel(f"Справа один и тот же фрагмент записи {int(row.track_id):06d} ({row.genre}). Пересчёт выполнен только для рисунка; файлы не изменены.", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mono_comparison_fragment(row, data_dir: Path):
    sample_rate = int(row.sample_rate)
    audio, _ = load_mono_audio(data_dir / row.audio_path)
    audio = audio[10 * sample_rate:10 * sample_rate + sample_rate // 5]
    size = sample_rate // 25
    step = sample_rate // 100
    starts = np.arange(0, len(audio) - size + 1, step)
    powers = [float(np.mean(np.square(audio[start:start + size], dtype=np.float64))) for start in starts]
    start = int(starts[np.argmax(powers)])
    fragment = audio[start:start + size]
    peak = float(np.max(np.abs(fragment)))
    if peak < 1e-5:
        return None
    normalized = fragment / peak
    spectrum = np.abs(np.fft.rfft(normalized * np.hanning(len(normalized)))) ** 2
    frequencies = np.fft.rfftfreq(len(normalized), 1 / sample_rate)
    edges = (0, 250, 500, 1000, 2000, 4000, 8000, 16000, sample_rate / 2 + 1)
    bands = np.array([spectrum[(frequencies >= left) & (frequencies < right)].sum() for left, right in zip(edges[:-1], edges[1:])])
    bands /= max(float(bands.sum()), 1e-12)
    envelope = np.array([np.sqrt(np.mean(part ** 2)) for part in np.array_split(normalized, 4)])
    features = np.r_[bands, np.mean(np.diff(np.signbit(normalized))), np.sqrt(np.mean(normalized ** 2)), envelope / max(float(envelope.max()), 1e-12)]
    return dict(row=row, audio=fragment, start=10 + start / sample_rate, features=features)


def plot_mono_origin_comparisons(original: pd.DataFrame, tracks: pd.DataFrame, data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = original.loc[original["split"].eq("train") & original["sample_rate"].eq(TARGET_SAMPLE_RATE), ["track_id", "genre", "channels", "sample_rate"]].merge(tracks[["track_id", "audio_path"]], on="track_id", validate="one_to_one")
    native = rows.loc[rows["channels"].eq(1)]
    genres = native.groupby("genre").size().nlargest(3).index
    report = []
    for genre in genres:
        reference = None
        for row in native.loc[native["genre"].eq(genre)].sort_values("track_id").itertuples():
            reference = mono_comparison_fragment(row, data_dir)
            if reference is not None:
                break
        candidates = [mono_comparison_fragment(row, data_dir) for row in rows.loc[rows["channels"].eq(2) & rows["genre"].eq(genre)].sort_values("track_id").itertuples()]
        candidates = [item for item in candidates if item is not None]
        if reference is None or len(candidates) < 3:
            raise ValueError(f"Недостаточно активных фрагментов для сравнения: {genre}")
        features = np.array([item["features"] for item in candidates])
        anchors = [reference["features"]]
        selected = []
        for _ in range(3):
            distances = np.linalg.norm(features[:, None] - np.array(anchors)[None, :], axis=2).min(axis=1)
            distances[selected] = -1
            index = int(np.argmax(distances))
            selected.append(index)
            anchors.append(features[index])
        examples = [reference] + [candidates[index] for index in selected]
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, sharey=True, layout="constrained")
        limit = max(float(np.max(np.abs(item["audio"]))) for item in examples) * 1.05
        for index, (item, color) in enumerate(zip(examples, ("#242424", "#337c9e", "#d99547", "#8b5496"))):
            row, audio = item["row"], item["audio"]
            time = np.arange(len(audio)) / row.sample_rate * 1000
            origin = "Исходное моно" if row.channels == 1 else f"Моно из стерео {index}"
            width = 1.6 if index == 0 else 0.8
            axes[index].plot(time, audio, color=color, linewidth=width)
            axes[index].set(title=f"{origin} · {row.track_id:06d} · с {item['start']:.2f} с", ylabel="Амплитуда", ylim=(-limit, limit))
            report.append(dict(genre=genre, reference_track_id=reference["row"].track_id, track_id=row.track_id, source_channels=row.channels, fragment_start_seconds=item["start"], fragment_duration_seconds=len(audio) / row.sample_rate, sample_rate=row.sample_rate, distance_to_reference=float(np.linalg.norm(item["features"] - reference["features"]))))
        axes[-1].set_xlabel("Время от начала фрагмента, мс")
        for axis in axes:
            axis.set_xlim(0, 40)
            axis.grid(alpha=0.2)
        fig.suptitle(f"{genre} · исходное моно и три разных моно из стерео · {TARGET_SAMPLE_RATE} Гц", fontsize=15)
        path = output_dir / f"origin_{genre.lower()}_{reference['row'].track_id:06d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Сравнение моно: {path}", flush=True)
    report_dir = Path(__file__).resolve().parent / "reports/mono"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).to_csv(report_dir / "origin_comparisons.csv", index=False)


def create_example_plots(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(data_dir / "processed/tracks_clean.csv")
    tracks = pd.read_csv(data_dir / "processed/tracks_mono.csv")
    print("Подбираю различающиеся примеры из обучающей выборки...", flush=True)
    plot_genre_examples(tracks, data_dir, output_dir / "genre_examples.png")
    plot_sample_rates(original, tracks, data_dir, output_dir / "sample_rates.png")
    plot_mono_origin_comparisons(original, tracks, data_dir, output_dir / "mono")
    print(f"Графики примеров: {output_dir}", flush=True)


def plot_resampling_example(row, data_dir: Path, path: Path) -> None:
    before, source_rate = load_mono_audio(data_dir / row.audio_path)
    after, target_rate = resample_signal(before, source_rate)
    source_time = np.arange(len(before)) / source_rate
    target_time = np.arange(len(after)) / target_rate
    _, levels = rms_envelope(before, source_rate)
    start = (int(np.argmax(levels)) + 0.5) * (source_rate // 10) / source_rate
    end = start + 0.002
    source_mask = (source_time >= start) & (source_time < end)
    target_mask = (target_time >= start) & (target_time < end)
    fig, axis = plt.subplots(figsize=(13, 4.5), layout="constrained")
    axis.plot((source_time[source_mask] - start) * 1000, before[source_mask], "o-", color="#d99547", markersize=4, markerfacecolor="none", linewidth=0.8, label=f"До: {source_mask.sum()} отсчётов на интервале")
    axis.plot((target_time[target_mask] - start) * 1000, after[target_mask], "o-", color="#337c9e", markersize=3, linewidth=1, label=f"После: {target_mask.sum()} отсчётов на интервале")
    axis.set(title=f"Один и тот же фрагмент 2 мс · с {start:.2f} с записи", xlabel="Время от начала фрагмента, мс", ylabel="Амплитуда", xlim=(0, 2))
    axis.legend(loc="best", fontsize=9)
    axis.grid(alpha=0.2)
    action = "ресемплинг" if row.was_resampled else "без пересчёта"
    fig.suptitle(f"{row.track_id:06d} · {row.genre} · {source_rate} → {target_rate} Гц · {action}", fontsize=14)
    fig.supxlabel("Отсчёты сравниваются на одном временном интервале. Амплитуда не нормировалась.", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_resampling(tracks: pd.DataFrame, data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "resampling"
    examples_dir.mkdir(exist_ok=True)
    selected = tracks.loc[tracks["split"].eq("train")].sort_values("track_id").drop_duplicates("source_sample_rate").sort_values("source_sample_rate")
    for row in selected.itertuples():
        plot_resampling_example(row, data_dir, examples_dir / f"track_{row.track_id:06d}.png")
    print(f"Графики ресемплинга: {examples_dir}", flush=True)


def plot_trimming(tracks: pd.DataFrame, data_dir: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = tracks["frames_before_trim"]
    selected = tracks.loc[[lengths.idxmin(), (lengths - lengths.median()).abs().idxmin(), lengths.idxmax()]].drop_duplicates("track_id")
    sample_rate = int(tracks["sample_rate"].iloc[0])
    target_frames = int(tracks["frames"].iloc[0])
    start = max(0, target_frames - sample_rate // 20)
    max_cut_ms = float(tracks["cropped_frames"].max()) / sample_rate * 1000
    fig, axes = plt.subplots(len(selected), 1, figsize=(14, 9), sharex=True, squeeze=False, layout="constrained")
    for axis, row in zip(axes[:, 0], selected.itertuples()):
        full, _ = load_resampled_audio(data_dir / row.audio_path)
        before = full[start:]
        after = trim_signal(full, target_frames)[start:]
        before_time = (np.arange(len(before)) + start - target_frames) / sample_rate * 1000
        after_time = (np.arange(len(after)) + start - target_frames) / sample_rate * 1000
        axis.plot(before_time, before, color="#d99547", linewidth=0.8, alpha=0.8, label="До обрезки")
        axis.plot(after_time, after, color="#337c9e", linewidth=1, label="После обрезки")
        axis.axvline(0, color="#b44646", linestyle="--", linewidth=1, label="Граница обрезки")
        cut_ms = row.cropped_frames / sample_rate * 1000
        if row.cropped_frames:
            axis.axvspan(0, cut_ms, color="#f7d5cf", alpha=0.6, label="Удалённый конец")
        axis.set(title=f"{row.track_id:06d} · {row.genre} · удалено {row.cropped_frames} отсчётов ({cut_ms:.2f} мс)", ylabel="Амплитуда", xlim=(-50, max_cut_ms + 2))
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Время относительно границы обрезки, мс")
    fig.suptitle(f"Обрезка конца при загрузке до {target_frames / sample_rate:.8f} с · {TARGET_SAMPLE_RATE} Гц", fontsize=15)
    fig.legend(*axes[-1, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"График обрезки: {path}", flush=True)



def plot_audio_check(row, source, source_rate: int, processed, target_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = to_mono(source)
    source_time, source_rms = rms_envelope(mono, source_rate)
    target_time, target_rms = rms_envelope(processed, target_rate)
    active = int(np.argmax(target_rms))
    center = float(target_time[active])
    start, end = max(0.0, center - 0.02), center + 0.02
    source_start, source_end = round(start * source_rate), round(end * source_rate)
    target_start, target_end = round(start * target_rate), round(end * target_rate)
    source_fragment = mono[source_start:source_end]
    target_fragment = processed[target_start:target_end]
    limit = max(float(np.max(np.abs(source_fragment))), float(np.max(np.abs(target_fragment))), 1e-6) * 1.1
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout="constrained")
    axes[0].plot(source_time, source_rms, color="#d99547", linewidth=1.5, label="Исходный сигнал после усреднения каналов")
    axes[0].plot(target_time, target_rms, color="#337c9e", linewidth=1, linestyle="--", label="После ресемплинга и обрезки")
    axes[0].set(title="RMS-огибающие на общей шкале времени · окна 100 мс", xlabel="Время от начала записи, с", ylabel="RMS-амплитуда")
    axes[0].legend(fontsize=9)
    axes[1].plot((np.arange(len(source_fragment)) + source_start) / source_rate * 1000 - start * 1000, source_fragment, color="#d99547", linewidth=0.8, alpha=0.8, label=f"До: {source_rate} Гц")
    axes[1].plot((np.arange(len(target_fragment)) + target_start) / target_rate * 1000 - start * 1000, target_fragment, color="#337c9e", linewidth=0.8, linestyle="--", label=f"После: {target_rate} Гц")
    axes[1].set(title=f"Один и тот же активный фрагмент · с {start:.3f} с записи", xlabel="Время от начала фрагмента, мс", ylabel="Амплитуда", xlim=(0, 40), ylim=(-limit, limit))
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(f"{row.track_id:06d} · {row.genre} · {source.shape[1]} кан. / {source_rate} Гц → моно / {target_rate} Гц", fontsize=15)
    fig.supxlabel(f"Массивы: {source.shape} → {processed.shape}. Амплитуда для графика не нормируется; при ресемплинге отсчёты могут изменяться.", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_audio_check_plots(data_dir: Path, output_dir: Path) -> None:
    tracks = pd.read_csv(data_dir / "processed/tracks_trimmed.csv")
    for row in select_check_examples(tracks).itertuples():
        source, source_rate = load_audio(data_dir / row.audio_path)
        full, target_rate = resample_signal(to_mono(source), source_rate)
        processed = trim_signal(full, int(row.frames))
        plot_audio_check(row, source, source_rate, processed, target_rate, output_dir / f"track_{row.track_id:06d}.png")
    print(f"Графики проверки примеров: {output_dir}", flush=True)


def plot_overview(clean: pd.DataFrame, genres: pd.DataFrame, reasons: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    genres.set_index("genre")[["before", "after"]].rename( columns={"before": "До очистки", "after": "После очистки"} ).plot.barh(ax=axes[0, 0], color=["#b8c5d6", "#337c9e"], width=0.8)
    axes[0, 0].set(title="Записи по жанрам", xlabel="Количество записей", ylabel="")
    axes[0, 0].legend(title="", fontsize=9)
    clean["duration_seconds"].plot.hist(ax=axes[0, 1], bins=25, color="#337c9e", edgecolor="white")
    axes[0, 1].set(title="Длительность оставленных записей", xlabel="Секунды", ylabel="Количество записей")
    clean["sample_rate"].value_counts().sort_index().rename_axis("Частота, Гц").plot.bar( ax=axes[1, 0], color="#537b63", rot=0 )
    axes[1, 0].set(title="Исходные частоты дискретизации", ylabel="Количество записей")
    if not reasons.empty:
        labels = {
            "decoder_reported_error": "Ошибки декодера", "decode_error": "Не декодируется",
            "decode_error;decoder_reported_error": "Не декодируется + ошибки кадров",
            "duplicate_same_split": "Повторная запись", "duplicate_label_conflict": "Конфликт меток дубликатов",
        }
        axes[1, 1].barh([labels.get(reason, reason) for reason in reasons["reason"]], reasons["count"], color="#ba7866")
    else:
        axes[1, 1].text(0.5, 0.5, "Исключений нет", ha="center", va="center")
    axes[1, 1].set(title="Исключения: непересекающиеся группы причин", xlabel="Количество записей")
    fig.suptitle("FMA small: результаты подготовки данных", fontsize=16)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_mel_spectrogram(spectrogram, sample_rate: int, hop_length: int, path: Path, title: str, power_max: float | None = None) -> None:
    import librosa.display

    path.parent.mkdir(parents=True, exist_ok=True)
    limit = float(np.percentile(spectrogram, 99)) if power_max is None else power_max
    if power_max is None and limit == 0:
        limit = float(spectrogram.max())
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Верхняя граница мощности должна быть положительным конечным числом.")
    clipped = float(np.mean(spectrogram > limit)) * 100
    fig, axis = plt.subplots(figsize=(14, 6), layout="constrained")
    display = librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length, x_axis="time", y_axis="mel", cmap="magma", vmin=0, vmax=limit, ax=axis)
    axis.set(title=f"Мел-спектрограмма мощности · {title}", xlabel="Время от начала записи, с", ylabel="Частота, Гц · мел-шкала")
    fig.colorbar(display, ax=axis, label="Мощность", extend="max" if clipped > 0 else "neither")
    fig.supxlabel(f"{spectrogram.shape[0]} мел-полос × {spectrogram.shape[1]} временных кадров · {sample_rate} Гц · шаг {hop_length} отсчётов\nЦветовая шкала: 0–{limit:.2f}. Выше границы: {clipped:.2f}% значений, показаны максимальным цветом. Максимум массива: {spectrogram.max():.2f}.", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mel_scale_comparison(spectrogram, sample_rate: int, hop_length: int, path: Path, title: str) -> None:
    import librosa.display

    path.parent.mkdir(parents=True, exist_ok=True)
    from log_spectrograms import TOP_DB, to_log_mel

    log_power = to_log_mel(spectrogram)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, sharey=True, layout="constrained")
    linear = librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length, x_axis="time", y_axis="mel", cmap="magma", vmin=0, vmax=float(spectrogram.max()), ax=axes[0])
    axes[0].set(title=f"Линейная шкала · полный диапазон мощности 0–{spectrogram.max():.2f}", ylabel="Частота, Гц · мел-шкала")
    fig.colorbar(linear, ax=axes[0], label="Мощность")
    logarithmic = librosa.display.specshow(log_power, sr=sample_rate, hop_length=hop_length, x_axis="time", y_axis="mel", cmap="magma", ax=axes[1])
    axes[1].set(title=f"Логарифмическая мощность · ref=1 · top_db={TOP_DB:g}", xlabel="Время от начала записи, с", ylabel="Частота, Гц · мел-шкала")
    fig.colorbar(logarithmic, ax=axes[1], label="Мощность, дБ")
    axes[0].set_xlabel("")
    fig.suptitle(f"Одна мел-спектрограмма · две цветовые шкалы · {title}", fontsize=15)
    fig.supxlabel(f"дБ = 10 × log₁₀(P / 1) · amin=10⁻¹⁰. Значения ниже максимума на {TOP_DB:g} дБ приравнены к нижнему порогу.", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_log_mel_spectrogram(spectrogram, sample_rate: int, hop_length: int, path: Path, title: str) -> None:
    import librosa.display
    from log_spectrograms import REFERENCE_POWER, TOP_DB

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(14, 6), layout="constrained")
    display = librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length, x_axis="time", y_axis="mel", cmap="magma", ax=axis)
    axis.set(title=f"Логарифмическая мел-спектрограмма · {title}", xlabel="Время от начала записи, с", ylabel="Частота, Гц · мел-шкала")
    fig.colorbar(display, ax=axis, label=f"Мощность, дБ относительно {REFERENCE_POWER:g}")
    fig.supxlabel(f"{spectrogram.shape[0]} мел-полос × {spectrogram.shape[1]} кадров · {sample_rate} Гц · шаг {hop_length}\nЗначения ниже максимума на {TOP_DB:g} дБ приравнены к нижнему порогу.", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_log_mel_examples(examples: list, path: Path) -> None:
    import librosa.display
    from create_spectrograms import HOP_LENGTH
    from log_spectrograms import TOP_DB

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (len(examples) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, rows * 3.5), squeeze=False, sharex=True, sharey=True, layout="constrained")
    lower = min(float(example["spectrogram"].min()) for example in examples)
    upper = max(float(example["spectrogram"].max()) for example in examples)
    for axis, example in zip(axes.flat, examples):
        display = librosa.display.specshow(example["spectrogram"], sr=example["sample_rate"], hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", cmap="magma", vmin=lower, vmax=upper, ax=axis)
        axis.set(title=f"{example['genre']} · {example['track_id']:06d}", xlabel="Время, с", ylabel="Частота, Гц")
        axis.label_outer()
    for axis in list(axes.flat)[len(examples):]:
        axis.set_visible(False)
    fig.colorbar(display, ax=list(axes.flat), label="Мощность, дБ относительно 1", shrink=0.8)
    fig.suptitle("Лог-мел-спектрограммы · примеры восьми жанров из train", fontsize=16)
    fig.supxlabel(f"Общая цветовая шкала · ref=1 · top_db={TOP_DB:g} · по одной записи на жанр, не оценка всего набора", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_log_mel_diagnostics(examples: list, summary: pd.DataFrame, path: Path) -> None:
    from log_spectrograms import TOP_DB

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = min(4, len(examples))
    rows = (len(examples) + columns - 1) // columns
    fig = plt.figure(figsize=(16, rows * 3 + 3.5), layout="constrained")
    grid = fig.add_gridspec(rows + 1, columns)
    lower = min(float(example["spectrogram"].min()) for example in examples)
    upper = max(float(example["spectrogram"].max()) for example in examples)
    bins = np.linspace(lower, upper, 61)
    histogram_axes = []
    for index, (example, row) in enumerate(zip(examples, summary.itertuples(index=False))):
        axis = fig.add_subplot(grid[index // columns, index % columns])
        values = example["spectrogram"].ravel()
        axis.hist(values, bins=bins, weights=np.full(values.size, 100 / values.size), color="#337c9e")
        axis.axvline(row.cutoff_db, color="#ba594d", linestyle="--")
        axis.set(title=f"{example['genre']} · {example['track_id']:06d}\nПорог {row.cutoff_db:.1f} дБ · изменено {row.floored_percent:.2f}%", xlabel="Мощность, дБ относительно 1", ylabel="Доля элементов, %", xlim=(lower - 2, upper + 2))
        histogram_axes.append(axis)
    y_max = max(axis.get_ylim()[1] for axis in histogram_axes)
    for axis in histogram_axes:
        axis.set_ylim(0, y_max)
    axis = fig.add_subplot(grid[rows, :])
    labels = [f"{row.genre}\n{row.track_id:06d}" for row in summary.itertuples(index=False)]
    bars = axis.bar(labels, summary["floored_percent"], color="#ba7866")
    axis.bar_label(bars, fmt="%.2f%%", padding=4)
    axis.set(title=f"Доля элементов, изменённых порогом «максимум записи − {TOP_DB:g} дБ»", ylabel="Изменённые элементы, %", ylim=(0, max(1.0, float(summary["floored_percent"].max()) * 1.25)))
    fig.suptitle("Проверка логарифмирования · распределения и влияние нижнего порога", fontsize=16)
    fig.supxlabel("По одному примеру каждого жанра из train · одинаковые интервалы гистограмм и масштаб по доле элементов", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Построить графики подготовленных данных.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--kind", choices=("all", "waveforms", "examples", "resampling", "mono-origins", "trimming", "audio-check", "spectrogram", "log-spectrograms"), default="all", help="Группа графиков; по умолчанию все.")
    parser.add_argument("--power-max", type=float, help="Верхняя граница цветовой шкалы спектрограммы; по умолчанию 99-й перцентиль мощности.")
    parser.add_argument("--seed", type=int, help="Фиксировать случайный выбор записей для звуковых волн.")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = Path(__file__).resolve().parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.kind in ("all", "waveforms"):
        mono = pd.read_csv(data_dir / "processed/tracks_mono.csv")
        plot_waveforms(mono, data_dir, output_dir / "random_waveforms.png", args.seed)
    if args.kind in ("all", "examples"):
        create_example_plots(data_dir, output_dir)
    if args.kind in ("all", "resampling"):
        resampled = pd.read_csv(data_dir / "processed/tracks_resampled.csv")
        plot_resampling(resampled, data_dir, output_dir)
    if args.kind in ("all", "trimming"):
        trimmed = pd.read_csv(data_dir / "processed/tracks_trimmed.csv")
        plot_trimming(trimmed, data_dir, output_dir / "trimming_examples.png")
    if args.kind in ("all", "audio-check"):
        create_audio_check_plots(data_dir, output_dir / "audio_check")
    if args.kind in ("all", "spectrogram"):
        from create_spectrograms import create_spectrogram_example, HOP_LENGTH
        row, _, sample_rate, spectrogram = create_spectrogram_example(data_dir)
        path = output_dir / f"mel_spectrogram_{int(row.track_id):06d}.png"
        plot_mel_spectrogram(spectrogram, sample_rate, HOP_LENGTH, path, f"{int(row.track_id):06d} · {row.genre}", power_max=args.power_max)
        plot_mel_scale_comparison(spectrogram, sample_rate, HOP_LENGTH, output_dir / f"mel_scale_comparison_{int(row.track_id):06d}.png", f"{int(row.track_id):06d} · {row.genre}")
    if args.kind in ("all", "log-spectrograms"):
        from log_spectrograms import prepare_log_examples
        examples, summary = prepare_log_examples(data_dir)
        plot_log_mel_examples(examples, output_dir / "log_mel_examples.png")
        plot_log_mel_diagnostics(examples, summary, output_dir / "log_mel_diagnostics.png")
    if args.kind == "mono-origins":
        original = pd.read_csv(data_dir / "processed/tracks_clean.csv")
        mono = pd.read_csv(data_dir / "processed/tracks_mono.csv")
        plot_mono_origin_comparisons(original, mono, data_dir, output_dir / "mono")
    if args.kind == "all":
        metadata = pd.read_csv(data_dir / "processed/tracks_metadata.csv", keep_default_na=False)
        audit = pd.read_csv(data_dir / "processed/audio_audit.csv", keep_default_na=False)
        clean = pd.read_csv(data_dir / "processed/tracks_clean.csv", keep_default_na=False)
        genres = pd.concat([metadata["genre"].value_counts().rename("before"), clean["genre"].value_counts().rename("after")], axis=1).fillna(0).astype(int).rename_axis("genre").reset_index().sort_values("genre")
        reasons = audit.loc[audit["exclusion_reasons"].ne(""), "exclusion_reasons"].value_counts().rename_axis("reason").reset_index(name="count")
        plot_metadata(metadata, output_dir)
        plot_validation(audit, clean, output_dir)
        plot_mono_dataset(clean, mono, data_dir, output_dir / "mono")
        plot_overview(clean, genres, reasons, output_dir / "overview.png")
    print(f"Графики сохранены: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

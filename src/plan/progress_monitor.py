import atexit
import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import Dict, Optional

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover - UI optional
    tk = None  # type: ignore
    ttk = None  # type: ignore


@dataclass(frozen=True)
class ProgressPayload:
    overall_completed: int
    overall_total: int
    qc_counts: Dict[str, int]
    qc_target: int
    yard_completed: Dict[str, int]
    yard_di: Dict[str, int]
    yard_targets: Dict[str, int]


class PlannerProgressMonitor:
    def __init__(self) -> None:
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is not available on this system.")

        self._queue: mp.Queue = mp.Queue()
        self._ready = mp.Event()
        self._process = mp.Process(
            target=_ui_process,
            args=(self._queue, self._ready),
            daemon=True,
        )
        self._closed = False
        self._process.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("Timed out initialising progress monitor UI.")
        atexit.register(self._shutdown)

    def update_progress(
        self,
        overall_completed: int,
        overall_total: int,
        qc_counts: Dict[str, int],
        qc_target: int,
        yard_completed: Dict[str, int],
        yard_di: Dict[str, int],
        yard_targets: Dict[str, int],
    ) -> None:
        if not self._process.is_alive():
            return
        payload = ProgressPayload(
            overall_completed=overall_completed,
            overall_total=overall_total,
            qc_counts=dict(qc_counts),
            qc_target=qc_target,
            yard_completed=dict(yard_completed),
            yard_di=dict(yard_di),
            yard_targets=dict(yard_targets),
        )
        try:
            self._queue.put_nowait(payload)
        except queue.Full:  # pragma: no cover - defensive
            pass

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        try:
            self._queue.close()
        except Exception:
            pass
        try:
            self._queue.join_thread()
        except Exception:
            pass
        if self._process.is_alive():
            self._process.join(timeout=1.0)


def _ui_process(
    payload_queue: mp.Queue,
    ready: mp.Event,
) -> None:
    if tk is None or ttk is None:
        return

    root = tk.Tk()
    root.title("Planner Progress")

    # Palette
    BG_COLOR = "#050505"
    CARD_COLOR = "#0e0e10"
    BAR_TROUGH = "#1a1a1d"
    TEXT_PRIMARY = "#f5f5f5"
    TEXT_MUTED = "#9a9a9a"
    ACCENT_COLOR = "#A5ACC1"

    root.configure(padx=12, pady=12, bg=BG_COLOR)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("Card.TFrame", background=CARD_COLOR, relief="flat")
    style.configure("TFrame", background=CARD_COLOR)
    style.configure(
        "Heading.TLabel",
        background=CARD_COLOR,
        foreground=ACCENT_COLOR,
        font=("Helvetica", 13, "bold"),
    )
    style.configure(
        "Title.TLabel",
        background=CARD_COLOR,
        foreground=ACCENT_COLOR,
        font=("Helvetica", 13, "bold"),
    )
    style.configure(
        "Body.TLabel",
        background=CARD_COLOR,
        foreground=TEXT_PRIMARY,
        font=("Helvetica", 10),
    )
    style.configure(
        "Muted.TLabel",
        background=CARD_COLOR,
        foreground=TEXT_MUTED,
        font=("Helvetica", 10),
    )
    style.configure(
        "Info.TLabel",
        background=CARD_COLOR,
        foreground=TEXT_MUTED,
        font=("Helvetica", 9),
    )

    container = ttk.Frame(root, style="Card.TFrame", padding=(20, 20))
    container.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    def gradient_color(ratio: float) -> str:
        ratio = max(0.0, min(1.0, float(ratio)))
        stops = [
            (0.0, (217, 83, 79)),   # red
            (0.33, (240, 173, 78)), # orange
            (0.66, (253, 224, 71)), # yellow
            (1.0, (92, 184, 92)),   # green
        ]
        for idx in range(len(stops) - 1):
            start_pos, start_col = stops[idx]
            end_pos, end_col = stops[idx + 1]
            if ratio <= end_pos or idx == len(stops) - 2:
                local_span = end_pos - start_pos
                if local_span <= 0:
                    t = 0.0
                else:
                    t = (ratio - start_pos) / local_span
                r = int(round(start_col[0] + (end_col[0] - start_col[0]) * t))
                g = int(round(start_col[1] + (end_col[1] - start_col[1]) * t))
                b = int(round(start_col[2] + (end_col[2] - start_col[2]) * t))
                return f"#{r:02x}{g:02x}{b:02x}"
        r, g, b = stops[-1][1]
        return f"#{r:02x}{g:02x}{b:02x}"

    def brighten(color: str, factor: float) -> str:
        color = color.lstrip("#")
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        r = min(255, int(round(r + (255 - r) * factor)))
        g = min(255, int(round(g + (255 - g) * factor)))
        b = min(255, int(round(b + (255 - b) * factor)))
        return f"#{r:02x}{g:02x}{b:02x}"

    def update_bar_color(style_name: str, current: int, maximum: int) -> None:
        maximum = max(1, int(maximum))
        current = max(0, int(current))
        ratio = current / maximum
        color = gradient_color(ratio)
        light = brighten(color, 0.35 * ratio)
        dark = brighten(color, 0.12 * ratio)
        style.configure(
            style_name,
            background=color,
            lightcolor=light,
            darkcolor=dark,
            troughcolor=BAR_TROUGH,
            bordercolor=BAR_TROUGH,
        )

    overall_label_var = tk.StringVar(container, value="Overall: 0/0 Jobs")
    overall_label = ttk.Label(container, textvariable=overall_label_var, style="Heading.TLabel")
    overall_label.grid(row=0, column=0, sticky="w")

    overall_style = "ProgressOverall.Horizontal.TProgressbar"
    style.configure(
        overall_style,
        troughcolor=BAR_TROUGH,
        background=gradient_color(0.0),
        bordercolor=BAR_TROUGH,
        lightcolor=gradient_color(0.0),
        darkcolor=gradient_color(0.0),
    )
    overall_bar = ttk.Progressbar(container, length=360, mode="determinate", style=overall_style)
    overall_bar.grid(row=1, column=0, pady=(8, 18), sticky="we")

    qc_header_frame = ttk.Frame(container, style="Card.TFrame")
    qc_header_frame.grid(row=2, column=0, sticky="we", pady=(0, 4))
    qc_header_frame.columnconfigure(0, weight=1)
    qc_label = ttk.Label(qc_header_frame, text="QC Progress", style="Title.TLabel")
    qc_label.grid(row=0, column=0, sticky="w")
    qc_heading = ttk.Label(qc_header_frame, text="QC Jobs Completed", style="Heading.TLabel")
    qc_heading.grid(row=0, column=1, sticky="e")

    qc_frame = ttk.Frame(container, style="Card.TFrame")
    qc_frame.grid(row=3, column=0, sticky="we")
    qc_widgets: Dict[str, Dict[str, object]] = {}

    yard_header_frame = ttk.Frame(container, style="Card.TFrame")
    yard_header_frame.grid(row=4, column=0, sticky="we", pady=(16, 4))
    yard_header_frame.columnconfigure(0, weight=1)
    yard_label = ttk.Label(yard_header_frame, text="Yard Progress", style="Title.TLabel")
    yard_label.grid(row=0, column=0, sticky="w")
    yard_heading = ttk.Label(yard_header_frame, text="Yard Tasks Completed", style="Heading.TLabel")
    yard_heading.grid(row=0, column=1, sticky="e")

    yard_frame = ttk.Frame(container, style="Card.TFrame")
    yard_frame.grid(row=5, column=0, sticky="we")
    yard_widgets: Dict[str, Dict[str, object]] = {}

    container.columnconfigure(0, weight=1)
    qc_frame.columnconfigure(1, weight=1)
    yard_frame.columnconfigure(1, weight=1)

    def apply_payload(payload: ProgressPayload) -> None:
        overall_maximum = max(1, payload.overall_total)
        overall_value = min(payload.overall_completed, payload.overall_total)
        overall_bar["maximum"] = overall_maximum
        overall_bar["value"] = overall_value
        update_bar_color(overall_style, overall_value, overall_maximum)
        ratio_pct = (overall_value / overall_maximum) * 100 if overall_maximum else 0.0
        overall_label_var.set(
            f"Overall: {payload.overall_completed:,}/{payload.overall_total:,} Jobs ({ratio_pct:.1f}%)"
        )

        for idx, (qc_name, count) in enumerate(sorted(payload.qc_counts.items()), start=0):
            widgets = qc_widgets.get(qc_name)
            if widgets is None:
                label = ttk.Label(qc_frame, text=qc_name, width=10, anchor="w", style="Body.TLabel")
                label.grid(row=idx, column=0, padx=(0, 8), pady=2, sticky="w")
                style_name = f"ProgressQC.{qc_name}.Horizontal.TProgressbar"
                style.configure(
                    style_name,
                    troughcolor=BAR_TROUGH,
                    background=gradient_color(0.0),
                    bordercolor=BAR_TROUGH,
                    lightcolor=gradient_color(0.0),
                    darkcolor=gradient_color(0.0),
                )
                bar = ttk.Progressbar(qc_frame, length=300, mode="determinate", style=style_name)
                bar.grid(row=idx, column=1, pady=2, sticky="we")
                text_var = tk.StringVar(qc_frame)
                info = ttk.Label(qc_frame, textvariable=text_var, width=12, anchor="e", style="Info.TLabel")
                info.grid(row=idx, column=2, padx=(8, 0), pady=2, sticky="e")
                unit = ttk.Label(qc_frame, text="Jobs", width=6, anchor="w", style="Info.TLabel")
                unit.grid(row=idx, column=3, pady=2, sticky="w")
                widgets = {
                    "label": label,
                    "bar": bar,
                    "info": info,
                    "text_var": text_var,
                    "style": style_name,
                    "unit": unit,
                }
                qc_widgets[qc_name] = widgets
            else:
                widgets["label"].grid(row=idx, column=0, padx=(0, 8), pady=2, sticky="w")  # type: ignore[index]
                widgets["bar"].grid(row=idx, column=1, pady=2, sticky="we")  # type: ignore[index]
                widgets["info"].grid(row=idx, column=2, padx=(8, 0), pady=2, sticky="e")  # type: ignore[index]
                widgets["unit"].grid(row=idx, column=3, pady=2, sticky="w")  # type: ignore[index]

            bar = qc_widgets[qc_name]["bar"]  # type: ignore[index]
            text_var = qc_widgets[qc_name]["text_var"]  # type: ignore[index]
            style_name = qc_widgets[qc_name]["style"]  # type: ignore[index]
            maximum = max(1, payload.qc_target)
            value = min(count, payload.qc_target)
            bar["maximum"] = maximum
            bar["value"] = value
            update_bar_color(style_name, value, maximum)
            text_var.set(f"{count:,}/{payload.qc_target:,}")

        all_yards = sorted(payload.yard_targets.keys())
        for idx, yard_name in enumerate(all_yards, start=0):
            completed = payload.yard_completed.get(yard_name, 0)
            di_count = payload.yard_di.get(yard_name, 0)
            widgets = yard_widgets.get(yard_name)
            if widgets is None:
                label = ttk.Label(yard_frame, text=yard_name, width=10, anchor="w", style="Body.TLabel")
                label.grid(row=idx, column=0, padx=(0, 8), pady=2, sticky="w")
                style_name = f"ProgressYard.{yard_name}.Horizontal.TProgressbar"
                style.configure(
                    style_name,
                    troughcolor=BAR_TROUGH,
                    background=gradient_color(0.0),
                    bordercolor=BAR_TROUGH,
                    lightcolor=gradient_color(0.0),
                    darkcolor=gradient_color(0.0),
                )
                bar = ttk.Progressbar(yard_frame, length=300, mode="determinate", style=style_name)
                bar.grid(row=idx, column=1, pady=2, sticky="we")
                progress_var = tk.StringVar(yard_frame)
                progress_info = ttk.Label(yard_frame, textvariable=progress_var, width=16, anchor="e", style="Muted.TLabel")
                progress_info.grid(row=idx, column=2, padx=(8, 0), pady=2, sticky="e")
                tasks_label = ttk.Label(yard_frame, text="Tasks", width=8, anchor="w", style="Info.TLabel")
                tasks_label.grid(row=idx, column=3, pady=2, sticky="w")
                di_var = tk.StringVar(yard_frame)
                di_info = ttk.Label(yard_frame, textvariable=di_var, width=12, anchor="e", style="Info.TLabel")
                di_info.grid(row=idx, column=4, padx=(8, 0), pady=2, sticky="e")
                widgets = {
                    "label": label,
                    "bar": bar,
                    "progress_info": progress_info,
                    "progress_var": progress_var,
                    "tasks_label": tasks_label,
                    "di_info": di_info,
                    "di_var": di_var,
                    "style": style_name,
                }
                yard_widgets[yard_name] = widgets
            else:
                widgets["label"].grid(row=idx, column=0, padx=(0, 8), pady=2, sticky="w")  # type: ignore[index]
                widgets["bar"].grid(row=idx, column=1, pady=2, sticky="we")  # type: ignore[index]
                widgets["progress_info"].grid(row=idx, column=2, padx=(8, 0), pady=2, sticky="e")  # type: ignore[index]
                widgets["tasks_label"].grid(row=idx, column=3, pady=2, sticky="w")  # type: ignore[index]
                widgets["di_info"].grid(row=idx, column=4, padx=(8, 0), pady=2, sticky="e")  # type: ignore[index]

            bar = yard_widgets[yard_name]["bar"]  # type: ignore[index]
            progress_var = yard_widgets[yard_name]["progress_var"]  # type: ignore[index]
            di_var = yard_widgets[yard_name]["di_var"]  # type: ignore[index]
            style_name = yard_widgets[yard_name]["style"]  # type: ignore[index]
            target = payload.yard_targets.get(yard_name, 0)
            maximum = max(1, target)
            value = min(completed, maximum)
            bar["maximum"] = maximum
            bar["value"] = value
            update_bar_color(style_name, value, maximum)
            progress_var.set(f"{completed:,}/{target:,}")
            di_var.set(f"{di_count:,} DI")

    def pump_queue() -> None:
        should_continue = True
        try:
            while True:
                payload = payload_queue.get_nowait()
                if payload is None:
                    should_continue = False
                    root.after(0, root.destroy)
                    break
                apply_payload(payload)
        except queue.Empty:
            pass
        finally:
            if should_continue:
                root.after(100, pump_queue)

    ready.set()
    root.after(100, pump_queue)
    root.mainloop()


_monitor: Optional[PlannerProgressMonitor] = None
_monitor_lock = mp.Lock()


def get_progress_monitor() -> PlannerProgressMonitor:
    global _monitor
    if tk is None or ttk is None:
        raise RuntimeError("Tkinter is not available on this system.")

    with _monitor_lock:
        if _monitor is None:
            _monitor = PlannerProgressMonitor()
    return _monitor

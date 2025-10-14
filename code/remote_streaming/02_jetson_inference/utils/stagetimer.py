# stagetimer.py
"""
StageTimer: a tiny, reusable measurement utility for per-frame pipelines.

It aggregates stage timings (e.g., capture/preprocess/infer/render/total),
prints periodic rolling averages (like `--measure`), and returns compact
status-bar suffixes for your UI.

What's new
----------
• Tracks *any* span name (not just cap/prep/infer/post/rend/tot)
• Fixed log order of main stages: cap → prep → infer → post → rend → tot
• Inline sub-stages after their main stage (sorted by avg desc):
  - cap_* after cap
  - prep_* after prep
  - infer_* after infer
  - post_* after post
  - rend_* after rend (before tot)
• Spans not starting with cap*/prep*/infer*/post*/rend*/tot* are printed on a new line
• `status_suffix_all()` still supports descending sort for last-frame spans

Quick example
-------------
from utils.stagetimer import StageTimer

timer = StageTimer(enabled=True, warmup=10, log_interval=2.0)

while True:
    timer.next_frame()
    with timer.span("tot"):
        with timer.span("cap"):
            img = inp.Capture(format="rgb8", timeout=1000)
        with timer.span("prep"):
            t = preprocess(img)
        with timer.span("infer"):
            logits = model(t)
        with timer.span("post"):
            with timer.span("post_argmax"):
                ...
            with timer.span("post_upsample"):
                ...
        with timer.span("rend"):
            out.Render(img)

    extra = timer.status_suffix(keys=("cap", "prep", "infer", "post", "tot"))
    out.SetStatus(f"MyApp | ~{smoothed_fps:4.1f} FPS{extra}")

    timer.end_frame()
    timer.log_this_interval(smoothed_fps, runtime_tag="torch FP16", min_ms=0.1)
"""

import time
from contextlib import contextmanager
from collections import defaultdict
from typing import Iterable, Optional


class StageTimer:
    """
    Drop-in timing aggregator for per-frame pipelines.

    Tracks durations per named stage (e.g., "cap", "prep", "infer", "post",
    "rend", "tot"), and supports arbitrary nested spans. Accumulates rolling
    averages after a warmup period, and periodically prints a compact summary.
    """

    def __init__(
        self,
        enabled: bool,
        warmup: int = 10,
        log_interval: float = 2.0,
        printer=print,
        include: Optional[Iterable[str]] = None,
    ):
        self.enabled = bool(enabled)
        self.warmup = int(warmup)
        self.log_interval = float(log_interval)
        self.printer = printer
        self.include = set(include) if include is not None else None  # None = include all

        self.frame_idx = 0
        self.last_dur = {}                 # last-frame durations (ms) for all spans seen this frame
        self.acc = defaultdict(float)      # rolling sum of durations (ms) per span
        self.acc_n = 0                     # number of frames accumulated
        self._t_log = time.time()

    def next_frame(self):
        """Begin a new frame and clear last-frame stage timings."""
        self.frame_idx += 1
        self.last_dur.clear()

    @contextmanager
    def span(self, name: str):
        """Context manager to time a named stage."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.last_dur[name] = dt_ms
            if self.enabled and self.frame_idx > self.warmup:
                if self.include is None or name in self.include:
                    self.acc[name] += dt_ms

    def status_suffix(self, keys=("cap", "prep", "infer", "post", "tot")) -> str:
        """
        Build a compact status-bar suffix of last-frame stage timings.
        """
        if not self.enabled or self.frame_idx <= self.warmup:
            return ""
        parts = [f"{k} {self.last_dur.get(k, 0.0):4.1f}ms" for k in keys if k in self.last_dur]
        return (" | " + " ".join(parts)) if parts else ""

    def status_suffix_all(
        self,
        prefix: Optional[str] = None,
        top_k: Optional[int] = None,
        min_ms: float = 0.0,
    ) -> str:
        """
        Status suffix for *all* last-frame spans, optionally filtered by prefix,
        sorted by descending time. Handy for showing detailed sub-stages.
        """
        if not self.enabled or self.frame_idx <= self.warmup:
            return ""
        items = [(k, v) for k, v in self.last_dur.items() if v >= min_ms]
        if prefix:
            items = [(k, v) for k, v in items if k.startswith(prefix)]
        items.sort(key=lambda kv: kv[1], reverse=True)
        if top_k is not None:
            items = items[:top_k]
        parts = [f"{k} {v:4.1f}ms" for k, v in items]
        return (" | " + " ".join(parts)) if parts else ""

    def end_frame(self):
        """
        Finish the current frame and update rolling-average counters.
        Call once per frame after you've measured all desired stages.
        """
        if self.enabled and self.frame_idx > self.warmup:
            self.acc_n += 1

    def log_this_interval(
        self,
        fps: float,
        runtime_tag: str,
        top_k: Optional[int] = None,
        min_ms: float = 0.0,
    ):
        """
        Log rolling-average timings and FPS at the configured `log_interval`.

        Printing rules (first line):
          cap → cap_* → prep → prep_* → infer → infer_* → post → post_* → rend → rend_* → tot
        Second line:
          spans that DO NOT start with cap*/prep*/infer*/post*/rend*/tot*
          (sorted by avg desc, optional top_k, min_ms threshold applies)
        """
        if not self.enabled or self.frame_idx <= self.warmup:
            return

        now = time.time()
        if not (now - self._t_log >= self.log_interval and self.acc_n > 0):
            return

        # Build avg per span (ms), apply min_ms
        avg = {k: (self.acc[k] / self.acc_n) for k in self.acc.keys()}
        avg = {k: v for k, v in avg.items() if v >= min_ms}

        main_order = ["cap", "prep", "infer", "post", "rend", "tot"]
        included_keys = set()
        line1_parts = []

        def fmt(k: str) -> str:
            return f"{k} {avg[k]:5.1f} ms"

        def add_stage_with_subs(stage: str):
            # main stage
            if stage in avg:
                line1_parts.append(fmt(stage))
                included_keys.add(stage)
            # sub-stages stage_* sorted by avg desc
            subs = [(k, v) for k, v in avg.items() if k.startswith(stage + "_")]
            subs.sort(key=lambda kv: kv[1], reverse=True)
            for k, _ in subs:
                line1_parts.append(fmt(k))
                included_keys.add(k)

        # cap -> cap_* -> prep -> prep_* -> infer -> infer_* -> post -> post_* -> rend -> rend_* -> tot
        for stage in main_order:
            if stage == "tot":
                # tot printed last, without tot_* expansion (by design)
                if stage in avg:
                    line1_parts.append(fmt(stage))
                    included_keys.add(stage)
            else:
                add_stage_with_subs(stage)

        # Second line: everything that DOESN'T start with these prefixes
        main_prefixes = ("cap", "prep", "infer", "post", "rend", "tot")
        others = [(k, v) for k, v in avg.items()
                  if k not in included_keys and not k.startswith(main_prefixes)]
        others.sort(key=lambda kv: kv[1], reverse=True)
        if top_k is not None:
            others = others[:top_k]
        other_parts = [f"{k} {v:5.1f} ms" for k, v in others]

        # Print
        self.printer(
            f"[measure] avg over {self.acc_n} frames ({self.log_interval:.1f}s) | "
            + " | ".join(line1_parts)
            + f" | ~{fps:4.1f} FPS ({runtime_tag})"
        )
        if other_parts:
            self.printer("           other | " + " | ".join(other_parts))

        # reset accumulators
        self.acc.clear()
        self.acc_n = 0
        self._t_log = now

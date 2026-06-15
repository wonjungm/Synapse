#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

import torch


PROFILE_DEFAULTS = {
    "twostream_b128": {
        "mode": "twostream",
        "size": 10752,
        "num_streams": 2,
        "loop_count": 16,
    },
    "strongplus_b64": {
        "mode": "twostream",
        "size": 12288,
        "num_streams": 2,
        "loop_count": 48,
    },
    "strong_b256": {
        "mode": "single",
        "size": 9216,
        "inner_loops": 32,
    },
    "strongscaled_b512": {
        "mode": "single_sleep",
        "size": 9216,
        "inner_loops": 8,
        "cycle_sleep_sec": 0.003,
    },
}


def _write_marker(path_str: str, payload: str) -> None:
    if not path_str:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _resolve_profile(args: argparse.Namespace) -> dict:
    cfg = dict(PROFILE_DEFAULTS[args.profile])

    if args.size is not None:
        cfg["size"] = int(args.size)
    if args.num_streams is not None:
        cfg["num_streams"] = int(args.num_streams)
    if args.loop_count is not None:
        cfg["loop_count"] = int(args.loop_count)
    if args.inner_loops is not None:
        cfg["inner_loops"] = int(args.inner_loops)
    if args.cycle_sleep_sec is not None:
        cfg["cycle_sleep_sec"] = float(args.cycle_sleep_sec)

    return cfg


def _warmup_twostream(cfg: dict, dev: torch.device) -> tuple[list[torch.cuda.Stream], list[tuple[torch.Tensor, torch.Tensor]]]:
    streams = [torch.cuda.Stream(device=dev) for _ in range(int(cfg["num_streams"]))]
    bufs = []
    size = int(cfg["size"])
    for _ in range(int(cfg["num_streams"])):
        a = torch.randn(size, size, device=dev, dtype=torch.float16)
        b = torch.randn(size, size, device=dev, dtype=torch.float16)
        bufs.append((a, b))

    with torch.inference_mode():
        for stream, (a, b) in zip(streams, bufs):
            with torch.cuda.stream(stream):
                _ = a @ b
        torch.cuda.synchronize(dev)

    return streams, bufs


def _warmup_single(cfg: dict, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    size = int(cfg["size"])
    a = torch.randn(size, size, device=dev, dtype=torch.float16)
    b = torch.randn(size, size, device=dev, dtype=torch.float16)
    with torch.inference_mode():
        _ = a @ b
        torch.cuda.synchronize(dev)
    return a, b


def _wait_for_release(args: argparse.Namespace) -> None:
    release_path = Path(args.release_file)
    print(f"[bgload] armed; waiting for release file: {release_path}", flush=True)
    _write_marker(
        args.ready_file,
        f"ready_at={time.time():.6f}\npid={os.getpid()}\nprofile={args.profile}\n",
    )
    while not release_path.exists():
        time.sleep(float(args.poll_sec))

    release_ts = time.time()
    if args.post_release_delay_sec > 0:
        print(
            f"[bgload] release observed at {release_ts:.3f}; "
            f"sleeping extra {args.post_release_delay_sec:.3f}s before load",
            flush=True,
        )
        time.sleep(float(args.post_release_delay_sec))

    _write_marker(
        args.started_file,
        f"started_at={time.time():.6f}\nrelease_file={release_path}\n",
    )


def _run_twostream(cfg: dict, streams: list[torch.cuda.Stream], bufs: list[tuple[torch.Tensor, torch.Tensor]], dev: torch.device, duration_sec: float) -> None:
    loop_count = int(cfg["loop_count"])
    size = int(cfg["size"])
    print(
        f"[bgload] starting twostream load "
        f"(size={size}, streams={len(streams)}, loops={loop_count}, duration={duration_sec}s)",
        flush=True,
    )
    end = time.time() + duration_sec
    with torch.inference_mode():
        while time.time() < end:
            for _ in range(loop_count):
                for stream, (a, b) in zip(streams, bufs):
                    with torch.cuda.stream(stream):
                        _ = a @ b
            torch.cuda.synchronize(dev)


def _run_single(cfg: dict, a: torch.Tensor, b: torch.Tensor, dev: torch.device, duration_sec: float, sleep_per_cycle: float) -> None:
    inner_loops = int(cfg["inner_loops"])
    size = int(cfg["size"])
    print(
        f"[bgload] starting single-stream load "
        f"(size={size}, inner_loops={inner_loops}, cycle_sleep={sleep_per_cycle}s, duration={duration_sec}s)",
        flush=True,
    )
    end = time.time() + duration_sec
    with torch.inference_mode():
        while time.time() < end:
            for _ in range(inner_loops):
                _ = a @ b
            torch.cuda.synchronize(dev)
            if sleep_per_cycle > 0:
                time.sleep(sleep_per_cycle)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prewarm a GPU background load and release it on an external signal."
    )
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS.keys()), required=True)
    parser.add_argument("--release-file", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--started-file", default="")
    parser.add_argument("--duration-sec", type=float, default=4000.0)
    parser.add_argument("--post-release-delay-sec", type=float, default=0.0)
    parser.add_argument("--poll-sec", type=float, default=0.1)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--num-streams", type=int, default=None)
    parser.add_argument("--loop-count", type=int, default=None)
    parser.add_argument("--inner-loops", type=int, default=None)
    parser.add_argument("--cycle-sleep-sec", type=float, default=None)
    args = parser.parse_args()

    cfg = _resolve_profile(args)
    mode = cfg["mode"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gated_bgload.py")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")

    print(
        f"[bgload] profile={args.profile}, mode={mode}, "
        f"visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )

    if mode == "twostream":
        streams, bufs = _warmup_twostream(cfg, dev)
    elif mode in {"single", "single_sleep"}:
        a, b = _warmup_single(cfg, dev)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    print("[bgload] warmup complete", flush=True)
    _wait_for_release(args)

    try:
        if mode == "twostream":
            _run_twostream(cfg, streams, bufs, dev, float(args.duration_sec))
        else:
            sleep_per_cycle = float(cfg.get("cycle_sleep_sec", 0.0))
            _run_single(cfg, a, b, dev, float(args.duration_sec), sleep_per_cycle)
    except KeyboardInterrupt:
        print("[bgload] interrupted; shutting down", flush=True)
        return 130

    print("[bgload] duration complete; exiting", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

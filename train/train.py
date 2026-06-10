from pathlib import Path
import argparse
import sys
from runpy import run_path


SCRIPT_MAP = {
    "fbg_fog": "fbg_fog_train.py",
    "trip": "fbg_fog_train.py",
    "single": "fbg_fog_train.py",
    "weargait": "weargait_train.py",
    "fusion": "baselines/fusion_train.py",
    "deepav": "baselines/deepav_train.py",
    "focal": "baselines/focal_train.py",
    "taca": "baselines/taca_train.py",
}


def main() -> None:
    parser = argparse.ArgumentParser("Project training dispatcher")
    parser.add_argument("--mode", choices=sorted(SCRIPT_MAP.keys()), default=None)
    parser.add_argument(
        "--dataset",
        choices=["fbg", "fog", "weargait"],
        default=None,
        help="Dataset shortcut: fbg/fog use fbg_fog, weargait uses weargait.",
    )
    args, remainder = parser.parse_known_args()

    mode = args.mode
    if mode is None:
        mode = "weargait" if args.dataset == "weargait" else "fbg_fog"

    if args.dataset == "weargait" and mode != "weargait":
        parser.error("--dataset weargait must use --mode weargait or omit --mode")
    if args.dataset in ("fbg", "fog") and mode == "weargait":
        parser.error("--mode weargait does not accept --dataset fbg/fog")

    train_root = Path(__file__).resolve().parent
    script_path = train_root / SCRIPT_MAP[mode]
    for path in (script_path.parent, train_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    child_args = list(remainder)
    if args.dataset in ("fbg", "fog"):
        child_args = ["--dataset", args.dataset, *child_args]

    sys.argv = [str(script_path), *child_args]
    run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()

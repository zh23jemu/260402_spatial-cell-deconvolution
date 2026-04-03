from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_deconv.data import inspect_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="输出数据集摘要。")
    parser.add_argument("--dataset", default="human_lymph_node")
    args = parser.parse_args()
    summary = inspect_dataset(args.dataset)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Run Case 1 -> Case 2 -> Case 3 for a single image."""

from __future__ import annotations

import os
import subprocess
import sys

from forensic_pipeline.paths import FORENSIC_TIMELINE_CSV, IMAGE_EVIDENCE_CSV, project_path


def derive_media_id(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]


def run_module(module_name: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", f"forensic_pipeline.{module_name}", *args]
    print(f"\n>>> Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{module_name} failed with exit code {result.returncode}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m forensic_pipeline.run_cases <image_path>")
        raise SystemExit(1)

    image_path = str(project_path(sys.argv[1]))

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        raise SystemExit(1)

    media_id = derive_media_id(image_path)

    print("===================================")
    print(" Multimodal Forensic Pipeline Start ")
    print("===================================")
    print(f"Image path : {image_path}")
    print(f"Media ID   : {media_id}")

    run_module("case_1", [image_path])
    run_module("case_2", [media_id, str(FORENSIC_TIMELINE_CSV)])
    run_module("case_3", [media_id, str(IMAGE_EVIDENCE_CSV)])

    print("\n===================================")
    print(" Pipeline completed successfully ")
    print("===================================")


if __name__ == "__main__":
    main()

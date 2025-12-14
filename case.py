"""
Pipeline runner:
Runs Case 1 -> Case 2 -> Case 3 sequentially for a single image.

Assumptions:
- case_1.py, case_2.py, case_3.py are in the same directory
- Each script is executable as: python case_X.py
- media_id is derived from image filename (without extension)

Usage:
  python run_all_cases.py path/to/image.webp
"""

import subprocess
import sys
import os


def derive_media_id(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]


def run_script(script_name: str, args: list[str]):
    cmd = [sys.executable, script_name] + args
    print(f"\n>>> Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_all_cases.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    media_id = derive_media_id(image_path)

    print("===================================")
    print(" Multimodal Forensic Pipeline Start ")
    print("===================================")
    print(f"Image path : {image_path}")
    print(f"Media ID   : {media_id}")

    # -------------------------------
    # Case 1: OCR embedded text check
    # -------------------------------
    run_script(
        "case_1.py",
        [image_path],
    )

    # ------------------------------------
    # Case 2: Associated SMS text analysis
    # ------------------------------------
    run_script(
        "case_2.py",
        [media_id],
    )

    # --------------------------------
    # Case 3: Final image-only decision
    # --------------------------------
    run_script(
        "case_3.py",
        [media_id],
    )

    print("\n===================================")
    print(" Pipeline completed successfully ")
    print("===================================")


if __name__ == "__main__":
    main()

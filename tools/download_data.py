"""Fetch the original retargeted LAFAN1 clips and matching G1 URDF/meshes."""

from pathlib import Path
import shutil

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tracking"
REPO_ID = "lvhaidong/LAFAN1_Retargeting_Dataset"


def main():
    source = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=["g1/*.csv", "robot_description/g1/g1_29dof_rev_1_0.urdf", "robot_description/g1/meshes/*"],
            cache_dir=ROOT / ".cache" / "huggingface",
        )
    )
    motions = DATA / "motions/lafan1_raw"
    motions.mkdir(parents=True, exist_ok=True)
    for file in (source / "g1").glob("*.csv"):
        shutil.copy2(file, motions / file.name)
    shutil.copytree(source / "robot_description" / "g1", DATA / "robot/g1_description", dirs_exist_ok=True)
    clips = list(motions.glob("*.csv"))
    if len(clips) != 40:
        raise RuntimeError(f"Expected 40 LAFAN1 clips, found {len(clips)} in {motions}")
    print(f"Downloaded {len(clips)} motion clips and the G1 description to {DATA}")


if __name__ == "__main__":
    main()

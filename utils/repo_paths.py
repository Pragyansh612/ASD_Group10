"""Repository-root-relative paths for data, checkpoints, and outputs."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def results_dir():
    path = os.path.join(REPO_ROOT, "results")
    os.makedirs(path, exist_ok=True)
    return path


def clip_embeddings_dir():
    path = os.environ.get(
        "CLIP_EMBED_DIR", os.path.join(REPO_ROOT, "clip_embeddings")
    )
    os.makedirs(path, exist_ok=True)
    return path


def col_data_root():
    return os.environ.get("COL_DATA_ROOT", os.path.join(REPO_ROOT, "ColData"))


def col_subdir(*parts):
    return os.path.join(col_data_root(), "col", *parts)


def col_pywork(name="pywork"):
    return col_subdir(name)


def col_pyframes():
    return col_subdir("pyframes")


def col_pycrop():
    return col_subdir("pycrop")


def col_scores(workdir="pywork"):
    return os.path.join(col_pywork(workdir), "scores.pckl")


def exps_path(*parts):
    return os.path.join(REPO_ROOT, "exps", *parts)


def face_detector_weights():
    return os.path.join(
        REPO_ROOT, "model", "faceDetector", "s3fd", "sfd_face.pth"
    )

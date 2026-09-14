import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS = PROJECT_ROOT / "artifacts"
CKPT_DIR = ARTIFACTS / "checkpoints"
RESULTS_DIR = ARTIFACTS / "results"
CACHE_DIR = ARTIFACTS / "embedding_cache"
LOGS_DIR = ARTIFACTS / "logs"
VIZ_DIR = ARTIFACTS / "visualizations"

AUDIO_ROOT = Path(os.environ.get("FLAME_AUDIO_ROOT", "PATH/TO/mlaad_v9_layer_stats"))

LABELS_LANG = DATA_DIR / "labels_lang.json"
FACTSHEETS = DATA_DIR / "factsheets.json"

LAYER_WEIGHTS_CKPT = Path(os.environ.get("FLAME_LAYER_WEIGHTS_CKPT", "PATH/TO/layer_weights.pt"))

TEXT_EMB_MODEL_DEFAULT = "intfloat/e5-large-v2"

SEED = 42
N_VARIANTS = 5
AUDIO_DIM = 1024
N_AUDIO_LAYERS = 24
VAL_FRAC = 0.10

SEP = "|"

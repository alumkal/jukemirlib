import torch as t
import os

VQVAE = None
TOP_PRIOR = None
CACHE_DIR = os.path.expanduser("~") + "/.cache/jukemirlib"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"

REMOTE_PREFIX = "https://openaipublic.azureedge.net/jukebox/models/"

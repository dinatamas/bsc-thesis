import importlib
import logging
from pathlib import Path
import sys

import torch

# ============================================================================
# General configuration.
# ============================================================================

# The seed is used to initialize pseudo random behaviour.
# This aids in the reproducibility of experiments.
SEED = 42

# Logging related settings.
LOGLEVEL = logging.INFO
LOGFORMATTER = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
LOGHANDLER = logging.StreamHandler(sys.stderr)

# Whether to use CUDA or not.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Vocabulary configuration.
# -------------------------

FREQUENCY_CUTOFF = 0  # Words rarer than this will be discarded.
MAX_VOCAB_SIZE = 500000

# ============================================================================
# Static dataset-specific files.
# ============================================================================

SELF_PATH = Path(__file__).resolve().parent
GRAMMAR_PATH = Path(SELF_PATH, 'grammar.txt').resolve()
PARSER_PATH = Path(SELF_PATH, 'parser.py').resolve()
UNPARSER_PATH = Path(SELF_PATH, 'unparser.py').resolve()

# ---------------------
# Generated components.
# ---------------------

try:
    importlib.invalidate_caches()
    PARSER_MODULE = importlib.import_module('datasets.mlforse.parser')
    PARSE = PARSER_MODULE.parse_syntax_tree
    UNPARSER_MODULE = importlib.import_module('datasets.mlforse.unparser')
    UNPARSE = UNPARSER_MODULE.unparse_syntax_tree
except ModuleNotFoundError:
    pass  # Assume that the parser and unparser will not be used.

# ============================================================================
# Execution mode configuration.
# ============================================================================

# -----------------------
# Preprocessing settings.
# -----------------------

# This sets how many examples can be sorted in RAM.
# The higher the number, the more memory will be consumed.
INMEMORY_SHUFFLE_COUNT = 1000

# ------------------
# Training settings.
# ------------------

# Termination conditions.
MAX_EPOCH = 1000

# Output control.
PRINT_EVERY = 1  # Ratio in [0, 1].

# Model hyperparameters.
LEARNING_RATE = 0.001
HIDDEN_SIZE = 256

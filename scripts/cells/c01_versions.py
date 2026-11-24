import warnings
warnings.filterwarnings("ignore")
import os, sys, json
import numpy as np, pandas as pd, torch, transformers, TTS
from IPython.display import Audio, display

os.environ['COQUI_TOS_AGREED'] = '1'
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import xtts_eval as xe

print('локально:', 'torch', torch.__version__, '| transformers', transformers.__version__, '| coqui-tts', TTS.__version__,
      '| device', xe.default_device())
print('Kaggle  :', open('outputs/kaggle/versions.txt').read().strip())

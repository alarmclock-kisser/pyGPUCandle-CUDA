# pyGPUCandle-CUDA
Python (3.11) script for stretching audio on CUDA (GPU), if drivers and environment are set up correctly. Otherwise on CPU, being way slower. Using pytorch-tensors. 

# About
This is V155 (english language release),
published on June 24th 2024.

# Drivers required
CUDA 12.4.1
cuDNN 9.2.0
pytorch 2.5.0.dev20240613+cu124
torchaudio 2.4.0.dev20240613+cu124
backend soundfile

# Libraries required
import os, argparse, threading, warnings

import torch, torchaudio, librosa
from torchaudio import functional as F
from torchaudio import transforms as T

import sounddevice as SD
import numpy as NP
import soundfile as SF

// see requirements.txt NOT ALL are actually required by minimum, this is just my personal setup

# Usage
When everything mentioned earlier is setup (preferrably in the latest version found in the requirements / listed here),

1.0) Change variables in the Python code:
  1.1) in the pyGPUCandleVxyz.py code, you'll have to change FIL to a path on your system, where there's 100% a valid audio file, so the program can fall back to processing that, instead of nothing. Useful for testing.
  1.2) you can also change PTH to a set standard output path of your choice, not necessary! You can also leave it as it is.
  1.3) check if the FOR list is correct for your backend. Mine (and the recommended) is "soundfile", that's why I go with FOR = (".mp3", ".wav", ".flac").
  1.4) changing the other variables is optional. The script only needs a valid sound file set to FIL.

2.0) run the script in your Python IDLE or whatever IDE you like. You can also call the .py in your CMD / Terminal (even with arguments / parameters !)

3.0) Arguments:
  3.1) -f / --file STRING 'path to your audio file'
  3.2) -o / --output STRING 'path to your output directory'
  3.3) -s / --stretch STRING 'stretch factor OR desired bpm'
  3.4) -l / --logging TRUE (optional) 'enable logging? set to enable (optional)'

4.0) You don't have to call it with arguments, you will be asked for the parameters, if no args or missing args are given, in a sequence.

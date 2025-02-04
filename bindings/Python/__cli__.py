import sys
import subprocess
from pathlib import Path


# support for plssvm-train including command line arguments
def train():
    exe_path = Path(__file__).parent / "plssvm-train"
    subprocess.run([str(exe_path)] + sys.argv[1:])


# support for plssvm-predict including command line arguments
def predict():
    exe_path = Path(__file__).parent / "plssvm-predict"
    subprocess.run([str(exe_path)] + sys.argv[1:])


# support for plssvm-scale including command line arguments
def scale():
    exe_path = Path(__file__).parent / "plssvm-scale"
    subprocess.run([str(exe_path)] + sys.argv[1:])

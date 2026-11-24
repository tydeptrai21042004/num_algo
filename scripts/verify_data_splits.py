#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.datasets import verify_splits
if __name__=='__main__':print('Verified split fingerprint:',verify_splits())

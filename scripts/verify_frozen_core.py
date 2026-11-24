#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.common import verify_core,protocol_sha256
if __name__=='__main__':print('Frozen core:',verify_core());print('60-attack protocol:',protocol_sha256())

#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.images import freeze
p=argparse.ArgumentParser();p.add_argument('--calibration',required=True);p.add_argument('--validation',required=True);p.add_argument('--out',required=True)
if __name__=='__main__':
    a=p.parse_args();print(freeze(a.out,a.calibration,a.validation))

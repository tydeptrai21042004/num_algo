#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.datasets import download_images
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=['lite','paper','stress'],default='paper');download_images(p.parse_args().profile)

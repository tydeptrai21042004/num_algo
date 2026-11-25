#!/usr/bin/env python3
"""Use the same compact12 ablation runner as the publication experiment."""
import sys
from run_12_images import main
if __name__=='__main__':
    sys.argv[1:1]=['--stage','ablation'];main()

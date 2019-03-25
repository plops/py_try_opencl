#!/usr/bin/env python3
"""opencl test code.
Usage:
  run_01_ocl [-vh]

Options:
  -h --help               Show this screen
  -v --verbose            Print debugging output
"""
# martin kielhorn 2019-03-26
import os
import sys
import docopt
import traceback
import numpy as np
import pathlib
import time
def current_milli_time():
    return int(round(((1000)*(time.time()))))
args=docopt.docopt(__doc__, version="0.0.1")
if ( args["--verbose"] ):
    print(args)
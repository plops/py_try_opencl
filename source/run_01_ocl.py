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
import pyopencl as cl
import pathlib
import time
def current_milli_time():
    return int(round(((1000)*(time.time()))))
class bcolors():
    OKGREEN="\033[92m"
    WARNING="\033[93m"
    FAIL="\033[91m"
    ENDC="\033[0m"
global g_last_timestamp
g_last_timestamp=current_milli_time()
def milli_since_last():
    global g_last_timestamp
    current_time=current_milli_time()
    res=((current_time)-(g_last_timestamp))
    g_last_timestamp=current_time
    return res
def fail(msg):
    print(((bcolors.FAIL)+("{:8d} FAIL ".format(milli_since_last()))+(msg)+(bcolors.ENDC)))
    sys.stdout.flush()
def plog(msg):
    print(((bcolors.OKGREEN)+("{:8d} LOG ".format(milli_since_last()))+(msg)+(bcolors.ENDC)))
    sys.stdout.flush()
args=docopt.docopt(__doc__, version="0.0.1")
if ( args["--verbose"] ):
    print(args)
platforms=cl.get_platforms()
platform=platforms[0]
devices=platform.get_devices(cl.device_type.GPU)
device=devices[0]
context=cl.Context([device])
plog("using first device of {}".format(devices))
queue=cl.CommandQueue(context, device)
plog("create refractive index distribution")
img_in_y=128
img_in_z=34
img_in=np.full((img_in_z,img_in_y,), (1.4999999999999997e+0), dtype=np.float32)
try:
    plog("instantiate in and output arrays on the gpu")
    gpu_shape=img_in.T.shape
    img_in_gpu=cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT), shape=gpu_shape)
    img_out_gpu=cl.Image(context, cl.mem_flags.WRITE_ONLY, cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT), shape=gpu_shape)
except Exception as e:
    type_, value_, tb_=sys.exc_info()
    lineno=tb_.tb_lineno
    for e in traceback.format_tb(tb_):
        fail(e)
    fmts=cl.get_supported_image_formats(context, cl.mem_flags.READ_ONLY, cl.mem_object_type.IMAGE2D)
    plog("supported READ_ONLY IMAGE2D formats: {}.".format(fmts))
plog("define opencl program.")
program_code="""__constant sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                                CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE);

__kernel void morph_op_kernel(__read_only image2d_t in, int op,
                              __write_only image2d_t out) {
  { const int x = get_global_id(0); }
}"""
program=cl.Program(context, program_code).build()
plog("built opencl program.")
kernel=cl.Kernel(program, "morph_op_kernel")
kernel.set_arg(0, img_in_gpu)
kernel.set_arg(1, np.uint32(0))
kernel.set_arg(2, img_out_gpu)
plog("defined opencl kernel.")
cl.enqueue_copy(queue, img_in_gpu, img_in, origin=(0,0,), region=gpu_shape, is_blocking=False)
cl.enqueue_nd_range_kernel(queue, kernel, gpu_shape, None)
plog("copied data to gpu and ran opencl kernel.")
img_out=np.empty_like(img_in)
cl.enqueue_copy(queue, img_out, img_out_gpu, origin=(0,0,), region=gpu_shape, is_blocking=True)
plog("copied gpu result back to cpu.")
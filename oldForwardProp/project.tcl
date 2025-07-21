############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
############################################################
open_project viv_hls_prj
set_top top
add_files cnn.cpp
add_files core.h
add_files conv.h
add_files pooling.h
add_files FullyConnected.h
add_files utils.h
add_files input_bpsk.h
add_files params.h
add_files model_impl.h
add_files model_consts.h

add_files TestBench.cpp
add_files -tb TestBench.cpp -cflags "-Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas"
open_solution "cnn_synth"
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 20 -name default
get_clock_uncertainty default
csim_design 
#-ldflags {-std=c++11} -clean -O -compiler gcc
#csynth_design
#cosim_design 
#-compiler gcc -ldflags {-std=c++11} -tool xsim

report_timing

export_design -rtl verilog -format ip_catalog -vendor "top" -ipname "top"
exit


#open_project viv_hls_prj/solution1/impl/verilog/top
#launch_runs impl_1 -to step_impl -script impl_1.tcl


#ifndef MODEL_H
#define MODEL_H

#include <ap_fixed.h>
typedef ap_fixed<32,10> fixedP;

#include <stdint.h> 
/* MODEL FIXED PARAMETERS */

const int INPUT_ROWS  = 1;
const int INPUT_COLS  = 1024;
const int INPUT_DEPTH = 2;
const int NUM_CLASSES = 24;
const int PADDING     = 0;
const int NUM_LAYERS = 20;
const int INPUT_SIZE  = 2048;
const int OUTPUT_SIZE = NUM_CLASSES;
const int NUM_PARAMETERS = 430832;

/* MODEL LAYERS */

/* LAYER 0 (CONV) */

const int INPUT_DEPTH_0 = 2;
const int INPUT_ROWS_0 = 1;
const int INPUT_COLS_0 = 1024;
const int NUM_KERNELS_0 = 12;
const int KERNEL_ROW_0 = 1;
const int KERNEL_COL_0 = 3;
const int CONV_OUTPUT_SIZE_0 = 12288;

fixedP outputConv1DLayer_0 [CONV_OUTPUT_SIZE_0];

/* LAYER 1 (ReLU) */

const int INPUT_DEPTH_1 = 12;
const int INPUT_ROWS_1 = 1;
const int INPUT_COLS_1 = 1024;
const int RELU_OUTPUT_SIZE_1 = 12288;

fixedP outputReLULayer_1 [RELU_OUTPUT_SIZE_1];

/* LAYER 2 (CONV) */

const int INPUT_DEPTH_2 = 12;
const int INPUT_ROWS_2 = 1;
const int INPUT_COLS_2 = 1024;
const int NUM_KERNELS_2 = 12;
const int KERNEL_ROW_2 = 1;
const int KERNEL_COL_2 = 3;
const int CONV_OUTPUT_SIZE_2 = 12288;

fixedP outputConv1DLayer_2 [CONV_OUTPUT_SIZE_2];

/* LAYER 3 (ReLU) */

const int INPUT_DEPTH_3 = 12;
const int INPUT_ROWS_3 = 1;
const int INPUT_COLS_3 = 1024;
const int RELU_OUTPUT_SIZE_3 = 12288;

fixedP outputReLULayer_3 [RELU_OUTPUT_SIZE_3];

/* LAYER 4 (MaxPooling1D) */

const int INPUT_DEPTH_4 = 12;
const int INPUT_ROWS_4 = 1;
const int INPUT_COLS_4 = 1024;
const int POOL_ROW_4 = 1;
const int POOL_COL_4 = 2;
const int MAX_POOL_OUTPUT_SIZE_4 = 6144;

fixedP outputMaxPool1DLayer_4 [MAX_POOL_OUTPUT_SIZE_4];

/* LAYER 5 (CONV) */

const int INPUT_DEPTH_5 = 12;
const int INPUT_ROWS_5 = 1;
const int INPUT_COLS_5 = 512;
const int NUM_KERNELS_5 = 24;
const int KERNEL_ROW_5 = 1;
const int KERNEL_COL_5 = 3;
const int CONV_OUTPUT_SIZE_5 = 12288;

fixedP outputConv1DLayer_5 [CONV_OUTPUT_SIZE_5];

/* LAYER 6 (ReLU) */

const int INPUT_DEPTH_6 = 24;
const int INPUT_ROWS_6 = 1;
const int INPUT_COLS_6 = 512;
const int RELU_OUTPUT_SIZE_6 = 12288;

fixedP outputReLULayer_6 [RELU_OUTPUT_SIZE_6];

/* LAYER 7 (CONV) */

const int INPUT_DEPTH_7 = 24;
const int INPUT_ROWS_7 = 1;
const int INPUT_COLS_7 = 512;
const int NUM_KERNELS_7 = 24;
const int KERNEL_ROW_7 = 1;
const int KERNEL_COL_7 = 3;
const int CONV_OUTPUT_SIZE_7 = 12288;

fixedP outputConv1DLayer_7 [CONV_OUTPUT_SIZE_7];

/* LAYER 8 (ReLU) */

const int INPUT_DEPTH_8 = 24;
const int INPUT_ROWS_8 = 1;
const int INPUT_COLS_8 = 512;
const int RELU_OUTPUT_SIZE_8 = 12288;

fixedP outputReLULayer_8 [RELU_OUTPUT_SIZE_8];

/* LAYER 9 (MaxPooling1D) */

const int INPUT_DEPTH_9 = 24;
const int INPUT_ROWS_9 = 1;
const int INPUT_COLS_9 = 512;
const int POOL_ROW_9 = 1;
const int POOL_COL_9 = 2;
const int MAX_POOL_OUTPUT_SIZE_9 = 6144;

fixedP outputMaxPool1DLayer_9 [MAX_POOL_OUTPUT_SIZE_9];

/* LAYER 10 (CONV) */

const int INPUT_DEPTH_10 = 24;
const int INPUT_ROWS_10 = 1;
const int INPUT_COLS_10 = 256;
const int NUM_KERNELS_10 = 32;
const int KERNEL_ROW_10 = 1;
const int KERNEL_COL_10 = 3;
const int CONV_OUTPUT_SIZE_10 = 8192;

fixedP outputConv1DLayer_10 [CONV_OUTPUT_SIZE_10];

/* LAYER 11 (ReLU) */

const int INPUT_DEPTH_11 = 32;
const int INPUT_ROWS_11 = 1;
const int INPUT_COLS_11 = 256;
const int RELU_OUTPUT_SIZE_11 = 8192;

fixedP outputReLULayer_11 [RELU_OUTPUT_SIZE_11];

/* LAYER 12 (CONV) */

const int INPUT_DEPTH_12 = 32;
const int INPUT_ROWS_12 = 1;
const int INPUT_COLS_12 = 256;
const int NUM_KERNELS_12 = 32;
const int KERNEL_ROW_12 = 1;
const int KERNEL_COL_12 = 3;
const int CONV_OUTPUT_SIZE_12 = 8192;

fixedP outputConv1DLayer_12 [CONV_OUTPUT_SIZE_12];

/* LAYER 13 (ReLU) */

const int INPUT_DEPTH_13 = 32;
const int INPUT_ROWS_13 = 1;
const int INPUT_COLS_13 = 256;
const int RELU_OUTPUT_SIZE_13 = 8192;

fixedP outputReLULayer_13 [RELU_OUTPUT_SIZE_13];

/* LAYER 14 (MaxPooling1D) */

const int INPUT_DEPTH_14 = 32;
const int INPUT_ROWS_14 = 1;
const int INPUT_COLS_14 = 256;
const int POOL_ROW_14 = 1;
const int POOL_COL_14 = 2;
const int MAX_POOL_OUTPUT_SIZE_14 = 4096;

fixedP outputMaxPool1DLayer_14 [MAX_POOL_OUTPUT_SIZE_14];

/* LAYER 15 (Flatten) */

const int INPUT_DEPTH_15 = 32;
const int INPUT_ROWS_15 = 1;
const int INPUT_COLS_15 = 128;
const int FLATTEN_OUTPUT_SIZE_15 = 4096;

fixedP outputFlattenLayer_15 [FLATTEN_OUTPUT_SIZE_15];

/* LAYER 15 (FC) */

const int INPUT_SIZE_15 = 4096;
const int NUM_NEURONS_15 = 100;

fixedP outputFCLayer_15 [NUM_NEURONS_15];

/* LAYER 16 (ReLU) */

const int INPUT_DEPTH_16 = 1;
const int INPUT_ROWS_16 = 1;
const int INPUT_COLS_16 = 100;
const int RELU_OUTPUT_SIZE_16 = 100;

fixedP outputReLULayer_16 [RELU_OUTPUT_SIZE_16];

/* LAYER 17 (FC) */

const int INPUT_SIZE_17 = 100;
const int NUM_NEURONS_17 = 100;

fixedP outputFCLayer_17 [NUM_NEURONS_17];

/* LAYER 18 (ReLU) */

const int INPUT_DEPTH_18 = 1;
const int INPUT_ROWS_18 = 1;
const int INPUT_COLS_18 = 100;
const int RELU_OUTPUT_SIZE_18 = 100;

fixedP outputReLULayer_18 [RELU_OUTPUT_SIZE_18];

/* LAYER 19 (FC) */

const int INPUT_SIZE_19 = 100;
const int NUM_NEURONS_19 = 24;

fixedP outputFCLayer_19 [NUM_NEURONS_19];



const int NUM_WORDS     = ______;
const int DDR_OFFSET_0  = 0;                                     // Flatten layer output stored here
const int DDR_OFFSET_1  = DDR_OFFSET_0  + FLATTEN_OUTPUT_SIZE_15 // FC Layer 1 output stored here
const int DDR_OFFSET_2  = DDR_OFFSET_1  + NUM_NEURONS_15;       // ReLU layer 1 output stored here
const int DDR_OFFSET_3  = DDR_OFFSET_2  + RELU_OUTPUT_SIZE_16; // FC Layer 2 output stored here
const int DDR_OFFSET_4  = DDR_OFFSET_3  + NUM_NEURONS_17;      // ReLu layer 2 output stored here 
const int DDR_OFFSET_5  = DDR_OFFSET_4  + RELU_OUTPUT_SIZE_18; // Final layer output stored here

#endif
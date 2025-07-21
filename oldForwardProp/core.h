#ifndef CORE_H
#define CORE_H

//#define IS_PIPELINED

#ifndef __SYNTHESIS__
//#include "/tools/Xilinx/Vivado/2018.3/include/gmp.h"
#include <gmp.h> 
#endif


#include "/home/fpessia/Documents/Vitis_Libraries/vision/L1/include/common/xf_video_mem.hpp"



#include <hls_stream.h>
#include <hls_math.h>
#include "ap_axi_sdata.h"

#include "conv.h"
#include "pooling.h"
#include "FullyConnected.h"


#include <ap_fixed.h>
typedef  ap_fixed<32, 10> fixedP;

enum Activation {
	LINEAR,
	SOFTMAX,
	SIGMOID,
	RELU
};

template<int KERNEL_ROW, int KERNEL_COL>
fixedP sum_window(xf::cv::Window<KERNEL_ROW,KERNEL_COL,fixedP>* window) {
	fixedP acc = 0;
	for (int r = 0; r < KERNEL_ROW; r++)

#ifdef IS_PIPELINED
		#pragma HLS PIPELINE
#endif

		for (int c = 0; c < KERNEL_COL; c++)

			acc +=(fixedP)window -> getval(r, c);

	return acc;
}



/*
template<int INPUT_ROWS, int INPUT_COLS, int POOL_ROW, int POOL_COL>
fixedP maxWindow1D(
		fixedP* inputData,
		int startCol);


template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int NUM_KERNELS, int KERNEL_ROW, int KERNEL_COL, int PADDING>
void convolveWithOneDimension1D(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* params,
		int idxDepth
	);


template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int NUM_KERNELS, int KERNEL_ROW, int KERNEL_COL, int PADDING>
void convolveWithOneKernel1D(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* params,
		fixedP* bias
	);



template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int NUM_KERNELS, int KERNEL_ROW, int KERNEL_COL, int PADDING>
void convolutionalLayer1D(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* params
	);



template<int INPUT_ROWS, int INPUT_COLS, int POOL_DIM>
fixedP maxWindow(
		fixedP* inputData,
		int startRow,
		int startCol);


template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int NUM_KERNELS, int KERNEL_DIM, int PADDING>
void convolveWithOneDimension(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* params,
		int idxDepth
	);


template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int NUM_KERNELS, int KERNEL_DIM, int PADDING>
void convolveWithOneKernel(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* params,
		fixedP* bias
	) ;




template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int NUM_KERNELS, int KERNEL_DIM, int PADDING>
void convolutionalLayer2D(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* params
	) ;



template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int POOL_ROW, int POOL_COL>
void maxPooling1D(
		fixedP* inputData,
		fixedP* outputData
	) ;



template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int POOL_DIM>
void maxPooling2D(
		fixedP* inputData,
		fixedP* outputData
	);

*/


template<int NUM_NEURONS, int ACTIVATION>
void computeActivation(
		fixedP* outputData) {

	switch(ACTIVATION) {

	case LINEAR:
		
		break;

	case SOFTMAX:
	{
		break;
	}

	case SIGMOID:
	{
		break;
	}

	default:
		break;
	}

}


/*
template<int INPUT_SIZE, int NUM_NEURONS, int ACTIVATION>
void fullyConnectedLayer(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* params
	); 


template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS>
void flattenLayerKeras(
		fixedP* inputData,
		fixedP* outputData
	);

template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS>
void flattenLayer(
		fixedP* inputData,
		fixedP* outputData
	); 


template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS>
void ReLU(
		fixedP* inputData,
		fixedP* outputData,
		fixedP alpha
	) {

	int outputsSoFar = 0;

	for (int idxDepth = 0; idxDepth < INPUT_DEPTH; idxDepth++){


		for (int idx = 0; idx < INPUT_ROWS * INPUT_COLS; idx++) {

#ifdef IS_PIPELINED
		#pragma HLS PIPELINE
#endif


			if(inputData[outputsSoFar] >= 0)
				outputData[outputsSoFar] = inputData[outputsSoFar];
			else
				outputData[outputsSoFar] = inputData[outputsSoFar] * alpha;
			outputsSoFar++;

		}

	}
}
*/

//template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS>
//void floatToFixedP(
//		float* inputData,
//		fixedP* outputData){
//
//		int outIdx = 0;
//
//		for(int idxDepth = 0; idxDepth < INPUT_DEPTH; idxDepth++){
//

//
//			for (int r  = 0; r  < INPUT_ROWS; r++) {
//				for (int c = 0; c < INPUT_COLS; c ++){
//					outputData[outIdx++] = (fixedP)inputData[idxDepth * INPUT_ROWS * INPUT_COLS + (r * INPUT_COLS) + c];
//				}
//			}
//		}
//}


//template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS>
//void fixedPToFloat(
//		fixedP* inputData,
//		float* outputData){
//
//		int outIdx = 0;
//
//		for(int idxDepth = 0; idxDepth < INPUT_DEPTH; idxDepth++){
//
//			#pragma HLS PIPELINE
//
//			for (int r  = 0; r  < INPUT_ROWS; r++) {
//				for (int c = 0; c < INPUT_COLS; c ++){
//					outputData[outIdx++] = (float)inputData[idxDepth * INPUT_ROWS * INPUT_COLS + (r * INPUT_COLS) + c];
//				}
//			}
//		}
//}
#endif

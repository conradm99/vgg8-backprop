#include "/home/miszczak-c/Documents/Vitis_Libraries/vision/L1/include/common/xf_video_mem.hpp"
//#include <gmp.h> 
#include <hls_stream.h>
#include <hls_math.h>
#include "ap_axi_sdata.h"
#include <ap_fixed.h>

//#define IS_PIPELINED

#include <ap_fixed.h>
typedef  ap_fixed<32,10> fixedP;



template<int INPUT_ROWS, int INPUT_COLS, int POOL_ROW, int POOL_COL>
fixedP maxWindow1D(
		fixedP* inputData,
		int startCol){

		fixedP max = -10^6;
		for (int c = startCol; c < startCol + POOL_COL; c++) {
			max = inputData[c] > max ? inputData[c] : max;
		}

		return max;
}

template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int POOL_ROW, int POOL_COL>
void maxPooling1D(
		fixedP* inputData,
		fixedP* outputData
	) {

	int outIdx = 0;

	for(int idxDepth = 0; idxDepth < INPUT_DEPTH; idxDepth++){

		fixedP* startPos = inputData + idxDepth * INPUT_COLS;
		/* Cannot pipeline */
		for (int c = 0; c < ((INPUT_COLS - POOL_COL)+1); c += POOL_COL){
			outputData[outIdx] = maxWindow1D<INPUT_ROWS, INPUT_COLS, POOL_ROW, POOL_COL>(startPos, c);
			outIdx++;
		}
	}

}


template<int INPUT_ROWS, int INPUT_COLS, int POOL_DIM>
fixedP maxWindow(
		fixedP* inputData,
		int startRow,
		int startCol){

		fixedP max = -10^6;
		for(int r = 0; r < INPUT_ROWS; r++) {

#ifdef IS_PIPELINED
		#pragma HLS PIPELINE
#endif

			for (int c = 0; c < INPUT_COLS; c++) {

				if (r >= startRow && r < startRow + POOL_DIM && c >= startCol && c < startCol + POOL_DIM)
					max = inputData[(r * INPUT_COLS) + c] > max ? inputData[(r * INPUT_COLS) + c] : max;
			}
		}
		return max;
}





template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int POOL_DIM>
void maxPooling2D(
		fixedP* inputData,
		fixedP* outputData
	) {

	int outIdx = 0;

	for(int idxDepth = 0; idxDepth < INPUT_DEPTH; idxDepth++){

		fixedP* startPos = inputData + idxDepth * INPUT_ROWS * INPUT_COLS;
		for (int r  = 0; r  < INPUT_ROWS; r+= POOL_DIM) {

			/* Cannot pipeline */
			for (int c = 0; c < INPUT_COLS; c += POOL_DIM){
				outputData[outIdx++] = maxWindow<INPUT_ROWS, INPUT_COLS, POOL_DIM>(startPos, r, c);
			}
		}
	}

}




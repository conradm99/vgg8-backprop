#include "/home/miszczak-c/Documents/Vitis_Libraries/vision/L1/include/common/xf_video_mem.hpp"
//#include <gmp.h> 
#include <hls_stream.h>
#include <hls_math.h>
#include "ap_axi_sdata.h"
#include <ap_fixed.h>

//#define IS_PIPELINED

#include <ap_fixed.h>
typedef  ap_fixed<32,10> fixedP;



template<int KERNEL_DIM>
fixedP sum_window_2d(xf::cv::Window<KERNEL_DIM,KERNEL_DIM,fixedP>* window) {
	fixedP acc = 0;
	for (int r = 0; r < KERNEL_DIM; r++)

#ifdef IS_PIPELINED
		#pragma HLS PIPELINE
#endif

		for (int c = 0; c < KERNEL_DIM; c++)

			acc +=(fixedP)window -> getval(r, c);

	return acc;
}


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


template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int NUM_KERNELS, int KERNEL_DIM, int PADDING>
void convolveWithOneDimension(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* params,
		int idxDepth
	)
{

	xf::cv::LineBuffer<KERNEL_DIM,INPUT_COLS + 2 * PADDING,fixedP> line_buf;
	xf::cv::Window<KERNEL_DIM,KERNEL_DIM,fixedP> window;

	int idxCol = 0;
	int idxRow = 0;
	int pix_convolved = 0;

	int numElemWritten = 0;
	int numElemRead	   = 0;

	int startRow = PADDING;
	int startCol = PADDING;
	int stopRow  = startRow + INPUT_ROWS - 1;
	int stopCol	 = startCol + INPUT_COLS - 1;

	bool needToWrite = false;

	/* Convolution of one kernel dimension with the respective input dimension */

	for (int idx_pixel = 0;
			 idx_pixel < ((INPUT_ROWS  + 2 * PADDING) * (INPUT_COLS  + 2 * PADDING));
			 idx_pixel++) {


#ifdef IS_PIPELINED
		#pragma HLS PIPELINE
#endif

		fixedP dataIn;

		if (idxRow >= startRow && idxRow <= stopRow && idxCol >= startCol && idxCol <= stopCol) {

			/* Need to select the right input given current depth */

			dataIn = inputData[numElemRead];
            numElemRead++;
		}
		else {
			dataIn = 0;
		}

		line_buf.shift_up(idxCol);
		line_buf.insert_top(dataIn, idxCol);

		/* Insert data in window and multiply by kernel */

		for (int idx_win_row = 0; idx_win_row < KERNEL_DIM; idx_win_row++) {

			#pragma HLS UNROLL

			for (int idx_win_col = 0; idx_win_col < KERNEL_DIM; idx_win_col++) {
				fixedP val = (fixedP) line_buf.getval(idx_win_row, idx_win_col + pix_convolved);

				//#pragma HLS UNROLL

				/* Need to select the right kernel given current dimension */

				val = params[(idx_win_row * KERNEL_DIM) + idx_win_col] * val;

				window.insert(val, idx_win_row, idx_win_col);
			}
		}

		/* Avoid computing out of image boundaries */

		fixedP val_outputStream = 0;
		if ((idxRow >= KERNEL_DIM - 1) && (idxCol >= KERNEL_DIM - 1)) {
			needToWrite = true;
			val_outputStream = sum_window_2d<KERNEL_DIM>(&window);
			pix_convolved++;
		}

		if(idxCol < (INPUT_COLS + 2 * PADDING) - 1)
			idxCol++;
		else {
			idxCol = 0;
			idxRow++;
			pix_convolved = 0;
		}

		if(needToWrite) {

			/* We increment the output at the right index.
			 * This will be done for each dimension
			 */
			if (!idxDepth) outputData[numElemWritten] = 0;
			outputData[numElemWritten] += val_outputStream;
            numElemWritten++;
			needToWrite = false;
		}
	}

}





template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int NUM_KERNELS, int KERNEL_DIM, int PADDING>
void convolveWithOneKernel(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* params,
		fixedP* bias
	) {




	for(int idxDepth = 0; idxDepth < INPUT_DEPTH; idxDepth++) {

		/* Doing one dimension at a time */

		convolveWithOneDimension
		<INPUT_DEPTH, INPUT_ROWS, INPUT_COLS, NUM_KERNELS, KERNEL_DIM, PADDING>
		(
			inputData  + idxDepth * INPUT_ROWS * INPUT_COLS,
			outputData,
			params + idxDepth * KERNEL_DIM * KERNEL_DIM, /* Each kernel has */
			idxDepth
		);

	}

	/* At the end, we can offset each output element by the bias */

	for (int idx = 0; idx < (INPUT_ROWS + 2 *PADDING  - KERNEL_DIM + 1)*(INPUT_COLS + 2*PADDING - KERNEL_DIM + 1); idx++) { 
		outputData[idx] += (fixedP)*bias;//INPUT_ROWS * INPUT_COLS
	}


}



template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int NUM_KERNELS, int KERNEL_DIM, int PADDING>
void convolutionalLayer2D(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* params
	) {

	for (int idxKernel = 0; idxKernel < NUM_KERNELS; idxKernel++) {

		convolveWithOneKernel
		<INPUT_DEPTH, INPUT_ROWS, INPUT_COLS, NUM_KERNELS, KERNEL_DIM, PADDING>
		(
			inputData,
			outputData + idxKernel*(INPUT_ROWS + 2 *PADDING  - KERNEL_DIM + 1)*(INPUT_COLS + 2*PADDING - KERNEL_DIM + 1), /* Each kernel produces */    //* INPUT_ROWS * INPUT_COLS
			params + idxKernel * INPUT_DEPTH * KERNEL_DIM * KERNEL_DIM, /* Each kernel has */
			params + NUM_KERNELS * INPUT_DEPTH * KERNEL_DIM * KERNEL_DIM + idxKernel
		);

	}
}







template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int NUM_KERNELS, int KERNEL_ROW, int KERNEL_COL, int PADDING>
void convolveWithOneDimension1D(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* params,
		int idxDepth
	)
{

	xf::cv::Window<KERNEL_ROW,KERNEL_COL,fixedP> window;

	int idxCol = 0;
	int idxRow = 0;
	int pix_convolved = 0;

	int numElemWritten = 0;
	int numElemRead	   = 0;

	int startCol = PADDING;
	int stopCol	 = startCol + INPUT_COLS - 1;


	/* Convolution of one kernel dimension with the respective input dimension */

	for (int idx_pixel = 0;
			 idx_pixel < (INPUT_COLS - KERNEL_COL  + 2 * PADDING) + 1;
			 idx_pixel++) {

#ifdef IS_PIPELINED
		#pragma HLS PIPELINE
#endif

		/* Insert data in window and multiply by kernel */

		for (int idx_win_row = 0; idx_win_row < KERNEL_ROW; idx_win_row++) {


			for (int idx_win_col = 0; idx_win_col < KERNEL_COL; idx_win_col++) {
				fixedP val;
				if(idx_win_col+pix_convolved >= startCol && idx_win_col+pix_convolved <= stopCol){

					val = inputData[idx_win_col + pix_convolved - startCol];

					/* Need to select the right kernel given current dimension */

					val = params[(idx_win_row * KERNEL_ROW) + idx_win_col] * val;
				}
				else{
					val =0;
				}


				window.insert(val, idx_win_row, idx_win_col);
			}
		}

		/* Avoid computing out of image boundaries */

		fixedP val_outputStream = 0;
		val_outputStream = sum_window<KERNEL_ROW, KERNEL_COL>(&window);
		pix_convolved++;
		outputData[numElemWritten] += val_outputStream;
		numElemWritten++;


	}
}


template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int NUM_KERNELS, int KERNEL_ROW, int KERNEL_COL, int PADDING>
void convolveWithOneKernel1D(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* params,
		fixedP* bias
	) {

	//Reset Output

	for (int i = 0; i < ((INPUT_COLS - KERNEL_COL  + 2 * PADDING) + 1); i++){
		outputData[i] = 0;
	}

	for(int idxDepth = 0; idxDepth < INPUT_DEPTH; idxDepth++) {

		/* Doing one dimension at a time */

		convolveWithOneDimension1D
		<INPUT_DEPTH, INPUT_ROWS, INPUT_COLS, NUM_KERNELS, KERNEL_ROW, KERNEL_COL, PADDING>
		(
			inputData  + idxDepth * INPUT_ROWS * INPUT_COLS,
			outputData,
			params + idxDepth * KERNEL_ROW * KERNEL_COL, /* Each kernel has */
			idxDepth
		);

	}

	/* At the end, we can offset each output element by the bias */

	for (int idx = 0; idx < ((INPUT_COLS - KERNEL_COL  + 2 * PADDING) + 1); idx++) {
		outputData[idx] += (fixedP)*bias;
	}
}


template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS, int NUM_KERNELS, int KERNEL_ROW, int KERNEL_COL, int PADDING>
void convolutionalLayer1D(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* params
	) {


	for (int idxKernel = 0; idxKernel < NUM_KERNELS; idxKernel++) {

		convolveWithOneKernel1D
		<INPUT_DEPTH, INPUT_ROWS, INPUT_COLS, NUM_KERNELS, KERNEL_ROW, KERNEL_COL, PADDING>
		(
			inputData,
			outputData + idxKernel * ((INPUT_COLS - KERNEL_COL  + 2 * PADDING) + 1), /* Each kernel produces */
			params + idxKernel * INPUT_DEPTH * KERNEL_ROW * KERNEL_COL, /* Each kernel has */
			params + NUM_KERNELS * INPUT_DEPTH * KERNEL_ROW * KERNEL_COL + idxKernel //biases
		);

	}

}



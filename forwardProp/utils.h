#ifndef UTILS_H
#define UTILS_H


#include <stdio.h>
/*
void print_conv_2d(fixedP* outputConvLayer, int ID, int K, int R, int C) {

	printf("Result of 2D Convolution, Layer: %d\n\n", ID);
	for(int idxKernel = 0; idxKernel < K; idxKernel++){
		printf("******* KERNEL #: %d *********\n\n", idxKernel);
		for(int idxRow = 0; idxRow <  R; idxRow++) {
			for (int idxCol = 0; idxCol < C; idxCol++) {
				printf(
					"%.15f\t",
					outputConvLayer
					[
						idxKernel * R * C +
						(idxRow   * C) + idxCol
					].to_float()
				);
			}
			printf("\n");
		}
		printf("\n");
	}
}

//void printFCLayer(float* outputFCLayer, int ID, int NUM_NEURONS) {
//	printf("Result of FC, Layer: %d\n\n", ID);
//	printf("*****************\n");
//	for (int idx = 0; idx < NUM_NEURONS; idx++){
//	  printf("%.15f\n", outputFCLayer[idx]);
//	}
//	printf("\n");
//}

void printFCLayer(fixedP* outputFCLayer, int ID, int NUM_NEURONS) {
	printf("Result of FC, Layer: %d\n\n", ID);
	printf("*****************\n");
	for (int idx = 0; idx < NUM_NEURONS; idx++){
	  printf("%.15f\n", outputFCLayer[idx].to_float());
	}
	printf("\n");
}

void printReLULayer(fixedP* outputReLULayer, int ID, int OUTPUT) {
	printf("Result of ReLU, Layer: %d\n\n", ID);
	printf("*****************\n");
	for (int idx = 0; idx < OUTPUT; idx++){
	  printf("%.15f\n", outputReLULayer[idx].to_float());
	}
	printf("\n");
}


//void printReLULayer(float* outputReLULayer, int ID, int D, int R, int C) {
//	printf("Result of ReLU, Layer: %d\n", ID);
//	printf("*****************\n");
//	for (int idxDepth = 0; idxDepth < D; idxDepth++){
//		for(int r = 0; r <  R; r++) {
//			for (int c = 0; c <  C; c++) {
//				printf("%.15f\t", outputReLULayer[idxDepth * R * C  + (r * C) + c]);
//			}
//			printf("\n");
//		}
//		printf("*******************\n");
//	}
//}


//maxPooling2D
//<INPUT_DEPTH_2, INPUT_ROWS_2, INPUT_COLS_2, POOL_DIM_2>
//(outputReLULayer_1, outputMaxPool2DLayer_2);

void printPoolingLayer(float* outputMaxPool2DLayer, int ID, int D, int R, int C, int P_FACT) {
		int R_AFT = R / P_FACT;
		int C_AFT = C / P_FACT;
		printf("Result of Pooling, Layer: %d\n\n", ID);

		for (int idxDepth = 0; idxDepth < D; idxDepth++){
			for(int r = 0; r < R_AFT; r++) {
				for (int c = 0; c < C_AFT; c++) {
					printf("%.15f\t", outputMaxPool2DLayer[idxDepth * R_AFT * C_AFT + (r * C_AFT ) + c]);
				}
				printf("\n");
			}
			printf("*****************\n");
		}
}


//void printFlattenLayer(float* outputFlattenLayer, int ID, int FLATTEN_SIZE) {
//		printf("Result of Flattening, Layer: %d\n\n", ID);
//		for (int idx = 0; idx < FLATTEN_SIZE; idx++) {
//			printf("%.15f\n", outputFlattenLayer[idx]);
//		}
//		printf("*****************\n\n");
//
//}


void printFlattenLayer(fixedP* outputFlattenLayer, int ID, int FLATTEN_SIZE) {
		printf("Result of Flattening, Layer: %d\n\n", ID);
		for (int idx = 0; idx < FLATTEN_SIZE; idx++) {
			printf("%.15f\n", outputFlattenLayer[idx].to_float());
		}
		printf("*****************\n\n");

}
*/
#endif

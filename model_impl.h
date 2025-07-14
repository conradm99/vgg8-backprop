convolutionalLayer1D
<INPUT_DEPTH_0, INPUT_ROWS_0, INPUT_COLS_0, NUM_KERNELS_0, KERNEL_ROW_0, KERNEL_COL_0,1>
(inputData, outputConv1DLayer_0, paramVector + 0);

ReLU
<INPUT_DEPTH_1, INPUT_ROWS_1, INPUT_COLS_1>
(outputConv1DLayer_0, outputReLULayer_1, 0 );

convolutionalLayer1D
<INPUT_DEPTH_2, INPUT_ROWS_2, INPUT_COLS_2, NUM_KERNELS_2, KERNEL_ROW_2, KERNEL_COL_2,1>
(outputReLULayer_1, outputConv1DLayer_2, paramVector + 84);

ReLU
<INPUT_DEPTH_3, INPUT_ROWS_3, INPUT_COLS_3>
(outputConv1DLayer_2, outputReLULayer_3, 0 );

maxPooling1D
<INPUT_DEPTH_4, INPUT_ROWS_4, INPUT_COLS_4, POOL_ROW_4, POOL_COL_4>
(outputReLULayer_3, outputMaxPool1DLayer_4);

convolutionalLayer1D
<INPUT_DEPTH_5, INPUT_ROWS_5, INPUT_COLS_5, NUM_KERNELS_5, KERNEL_ROW_5, KERNEL_COL_5,1>
(outputMaxPool1DLayer_4, outputConv1DLayer_5, paramVector + 528);

ReLU
<INPUT_DEPTH_6, INPUT_ROWS_6, INPUT_COLS_6>
(outputConv1DLayer_5, outputReLULayer_6, 0 );

convolutionalLayer1D
<INPUT_DEPTH_7, INPUT_ROWS_7, INPUT_COLS_7, NUM_KERNELS_7, KERNEL_ROW_7, KERNEL_COL_7,1>
(outputReLULayer_6, outputConv1DLayer_7, paramVector + 1416);

ReLU
<INPUT_DEPTH_8, INPUT_ROWS_8, INPUT_COLS_8>
(outputConv1DLayer_7, outputReLULayer_8, 0 );

maxPooling1D
<INPUT_DEPTH_9, INPUT_ROWS_9, INPUT_COLS_9, POOL_ROW_9, POOL_COL_9>
(outputReLULayer_8, outputMaxPool1DLayer_9);

convolutionalLayer1D
<INPUT_DEPTH_10, INPUT_ROWS_10, INPUT_COLS_10, NUM_KERNELS_10, KERNEL_ROW_10, KERNEL_COL_10,1>
(outputMaxPool1DLayer_9, outputConv1DLayer_10, paramVector + 3168);

ReLU
<INPUT_DEPTH_11, INPUT_ROWS_11, INPUT_COLS_11>
(outputConv1DLayer_10, outputReLULayer_11, 0 );

convolutionalLayer1D
<INPUT_DEPTH_12, INPUT_ROWS_12, INPUT_COLS_12, NUM_KERNELS_12, KERNEL_ROW_12, KERNEL_COL_12,1>
(outputReLULayer_11, outputConv1DLayer_12, paramVector + 5504);

ReLU
<INPUT_DEPTH_13, INPUT_ROWS_13, INPUT_COLS_13>
(outputConv1DLayer_12, outputReLULayer_13, 0 );

maxPooling1D
<INPUT_DEPTH_14, INPUT_ROWS_14, INPUT_COLS_14, POOL_ROW_14, POOL_COL_14>
(outputReLULayer_13, outputMaxPool1DLayer_14);

flattenLayer
<INPUT_DEPTH_15, INPUT_ROWS_15, INPUT_COLS_15>
(outputMaxPool1DLayer_14, outputFlattenLayer_15);

fullyConnectedLayer
<INPUT_SIZE_15, NUM_NEURONS_15, LINEAR>
(outputFlattenLayer_15, outputFCLayer_15, paramVector + 8608);

ReLU
<INPUT_DEPTH_16, INPUT_ROWS_16, INPUT_COLS_16>
(outputFCLayer_15, outputReLULayer_16, 0 );

fullyConnectedLayer
<INPUT_SIZE_17, NUM_NEURONS_17, LINEAR>
(outputReLULayer_16, outputFCLayer_17, paramVector + 418308);

ReLU
<INPUT_DEPTH_18, INPUT_ROWS_18, INPUT_COLS_18>
(outputFCLayer_17, outputReLULayer_18, 0 );

fullyConnectedLayer
<INPUT_SIZE_19, NUM_NEURONS_19, LINEAR>
(outputReLULayer_18, outputFCLayer_19, paramVector + 428408);

for (int idx = 0; idx < OUTPUT_SIZE; idx++)
	outputData[idx] = outputFCLayer_19[idx];


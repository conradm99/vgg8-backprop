memcpy(inputData_ping, (fixedP*)intermediateResultIF + DDR_OFFSET_5, NUM_NEURONS_19 * sizeof(fixedP));
memcpy(postAct_prev_ping, (fixedP*)intermediateResultIF + DDR_OFFSET_4, RELU_OUTPUT_SIZE_18 * sizeof(fixedP));
memcpy(weights_ping, (fixedP*)weightsBufferIF + weights_DDR_OFFSET0, NUM_WEIGHTS_2 * sizeof(fixedP));
memcpy(biases_ping, (fixedP*)biasBufferIF + bias_DDR_OFFSET1, NUM_BIASES_2 * sizeof(fixedP));
outputGradientBP<NUM_NEURONS_19, INPUT_SIZE_19>(
    inputData_ping,
    postAct_prev_ping,
    outputData_ping,
    //labels,
    weights_ping,
    biases_ping,
    lossOut_ping);
memcpy((fixedP*)gradients + gradients_DDR_OFFSET0, outputData_ping, NUM_GRADIENTS_2 * sizeof(fixedP));

// ----- Layer 10 (Linear) -----
memcpy(inputData_pong, (fixedP*)intermediateResultIF + intermediateResult_DDR_OFFSET10, NUM_NEURONS_17 * sizeof(fixedP));
memcpy(postAct_prev_pong, (fixedP*)intermediateResultIF + intermediateResult_DDR_OFFSET10, RELU_OUTPUT_SIZE_16 * sizeof(fixedP));
memcpy(weights_pong, (fixedP*)weightsBufferIF + weights_DDR_OFFSET1, NUM_WEIGHTS_1 * sizeof(fixedP));
memcpy(biases_pong, (fixedP*)biasBufferIF + bias_DDR_OFFSET1, NUM_BIASES_1 * sizeof(fixedP));
hiddenLayerBPFC<NUM_NEURONS_17, INPUT_SIZE_17, NUM_NEURONS_19>(
    inputData_pong,
    postAct_prev_pong,
    outputData_pong,
    weights_pong,
    biases_pong,
    lossOut_ping,
    lossOut_pong);
memcpy((fixedP*)gradients + gradients_DDR_OFFSET1, outputData_pong, NUM_GRADIENTS_1 * sizeof(fixedP));

// ----- Layer 9 (Linear) -----
memcpy(inputData_ping, (fixedP*)intermediateResultIF + intermediateResult_DDR_OFFSET9, NUM_NEURONS_15 * sizeof(fixedP));
memcpy(postAct_prev_ping, (fixedP*)intermediateResultIF + intermediateResult_DDR_OFFSET8, FLATTEN_OUTPUT_SIZE_15 * sizeof(fixedP));
memcpy(weights_ping, (fixedP*)weightsBufferIF + weights_DDR_OFFSET2, NUM_WEIGHTS_0 * sizeof(fixedP));
memcpy(biases_ping, (fixedP*)biasBufferIF + bias_DDR_OFFSET2, NUM_BIASES_0 * sizeof(fixedP));
hiddenLayerBPFC<NUM_NEURONS_15, INPUT_SIZE_15, NUM_NEURONS_17>(
    inputData_ping,
    postAct_prev_ping,
    outputData_ping,
    weights_ping,
    biases_ping,
    lossOut_pong,
    lossOut_ping);
memcpy((fixedP*)gradients + gradients_DDR_OFFSET2, outputData_ping, NUM_GRADIENTS_0 * sizeof(fixedP));

memcpy(weightGrads, (fixedP*)gradients, (NUM_WEIGHTS_0 + NUM_WEIGHTS_1 + NUM_WEIGHTS_2) * sizeof(fixedP));
memcpy(weightsBuffer, (fixedP*)weightsBufferIF, (NUM_WEIGHTS_0+NUM_WEIGHTS_1+NUM_WEIGHTS_2) * sizeof(fixedP));

paramUpdate<NUM_NEURONS_19, INPUT_SIZE_19>(weightsBuffer+weights_DDR_OFFSET0, LEARNING_RATE, weightGrads + gradients_DDR_OFFSET0);
paramUpdate<NUM_NEURONS_17, INPUT_SIZE_17>(weights_pong+weights_DDR_OFFSET1, LEARNING_RATE, weightGrads + gradients_DDR_OFFSET1);
paramUpdate<NUM_NEURONS_15, INPUT_SIZE_15>(weights_ping+weights_DDR_OFFSET2, LEARNING_RATE, weightGrads + gradients_DDR_OFFSET2);


//memcpy(dest,src,size)
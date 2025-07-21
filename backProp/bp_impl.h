memcpy(inputData_ping, (fixedP*)intermediateResultIF + DDR_OFFSET_5, NUM_NEURONS_19 * sizeof(fixedP));
memcpy(postAct_prev_ping, (fixedP*)intermediateResultIF + DDR_OFFSET_4, RELU_OUTPUT_SIZE_18 * sizeof(fixedP));
memcpy(weights_ping, (fixedP*)weightsBufferIF + weights_DDR_OFFSET0, 2400 * sizeof(fixedP));
memcpy(biases_ping, (fixedP*)biasBufferIF + bias_DDR_OFFSET1, 24 * sizeof(fixedP));
outputGradientBP<FULLYCONNECTED2_OUT_FEATURES, FULLYCONNECTED2_IN_FEATURES>(
    inputData_ping,
    postAct_prev_ping,
    outputData_ping,
    weights_ping,
    biases_ping,
    frontLayerLoss_ping,
    lossOut_ping);
memcpy((fixedP*)gradients + gradients_DDR_OFFSET0, outputData_ping, 24 * sizeof(fixedP));

// ----- Layer 10 (Linear) -----
memcpy(inputData_pong, (fixedP*)intermediateResultIF + intermediateResult_DDR_OFFSET10, 100 * sizeof(fixedP));
memcpy(postAct_prev_pong, (fixedP*)intermediateResultIF + intermediateResult_DDR_OFFSET10, 100 * sizeof(fixedP));
memcpy(weights_pong, (fixedP*)weightsBufferIF + weights_DDR_OFFSET1, 10000 * sizeof(fixedP));
memcpy(biases_pong, (fixedP*)biasBufferIF + bias_DDR_OFFSET1, 100 * sizeof(fixedP));
hiddenLayerBPFC<FULLYCONNECTED1_OUT_FEATURES, FULLYCONNECTED1_IN_FEATURES, 24>(
    inputData_pong,
    postAct_prev_pong,
    outputData_pong,
    weights_pong,
    biases_pong,
    frontLayerLoss_pong,
    lossOut_pong);
memcpy((fixedP*)gradients + gradients_DDR_OFFSET1, outputData_pong, 100 * sizeof(fixedP));

// ----- Layer 9 (Linear) -----
memcpy(inputData_ping, (fixedP*)intermediateResultIF + intermediateResult_DDR_OFFSET9, 4096 * sizeof(fixedP));
memcpy(postAct_prev_ping, (fixedP*)intermediateResultIF + intermediateResult_DDR_OFFSET8, 4096 * sizeof(fixedP));
memcpy(weights_ping, (fixedP*)weightsBufferIF + weights_DDR_OFFSET2, 409600 * sizeof(fixedP));
memcpy(biases_ping, (fixedP*)biasBufferIF + bias_DDR_OFFSET2, 100 * sizeof(fixedP));
hiddenLayerBPFC<FULLYCONNECTED0_OUT_FEATURES, FULLYCONNECTED0_IN_FEATURES, 100>(
    inputData_ping,
    postAct_prev_ping,
    outputData_ping,
    weights_ping,
    biases_ping,
    frontLayerLoss_ping,
    lossOut_ping);
memcpy((fixedP*)gradients + gradients_DDR_OFFSET2, outputData_ping, 100 * sizeof(fixedP));
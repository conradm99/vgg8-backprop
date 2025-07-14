#include <hls_stream.h>
#include <hls_math.h>
#include "ap_axi_sdata.h"
#include <ap_fixed.h> 

typedef ap_fixed<32,10> fixedP;

inline float HLSexponent(fixedP x) {
    // Convert fixed-point to float for the exponential operation
    float xf = static_cast<float>(x);
    // Calculate the exponential
    float exp_xf = hls::exp(xf);
    return exp_xf;
}



// working!
template<int INPUT_NUERONS>
void SoftMax(
        fixedP* inputData,
        float outputData[INPUT_NUERONS]
){
        float Buffer[INPUT_NUERONS];
        float Sum = 0.0;
        
        for(int n = 0 ; n < INPUT_NUERONS; n++){
            #pragma HLS PIPELINE
            Buffer[n] = HLSexponent(inputData[n]);
            Sum += Buffer[n];
        }

        for(int n = 0; n < INPUT_NUERONS; n++){
            #pragma HLS pipeline
            outputData[n] =  Buffer[n] / Sum;
        }
}

// working!
template<int ROWS, int COLS>
void paramUpdate(       // add vector sizes for this
    fixedP* param,      // weight/bias vector to update
    fixedP learningRate,
    fixedP* gradient
){
    fixedP updatedParam[ROWS*COLS];
    for (int i=0; i < ROWS*COLS; i++){
        updatedParam[i] = param[i] - learningRate * gradient[i];
    }
    for (int i = 0; i < ROWS*COLS; i++)
        printf("updated param: %f,\n", updatedParam[i].to_float());
}

// added loss output, need to test this 
template<int NUM_NEURONS, int INPUT_SIZE>
void outputGradientBP(                                      // Assumes output layer is a fully connected layer
    fixedP* inputData,                                      // input = predictions from forward propagation
    fixedP* postAct_prevlayer,                              // post activation output from the previous layer 
    fixedP* outputData,                                     // resulting data is a matrix of loss gradients, for example [84x10] in current lenet5 implementation 
    fixedP* labels,                                         // ground truth labels 
    fixedP* weights,
    fixedP* biases,
    fixedP* loss   
){
    float softmaxOutput[NUM_NEURONS];
    fixedP dL_dZ[NUM_NEURONS];                               // derivative of loss wrt current layers activation function
    
    SoftMax<NUM_NEURONS>(inputData, softmaxOutput);

    for (int idx = 0; idx < NUM_NEURONS; idx++){                // dL_dA calculation, for output layer this is just ypred - ylabel. there are NUM_NEURONs classes at the output, so that will be our index 
        #pragma HLS PIPELINE
        dL_dZ[idx] = (fixedP)softmaxOutput[idx] - labels[idx];     // Softmax output is a float so we cast it to fixedP
        loss[idx] = dL_dZ[idx];
    }
    for (int i = 0; i < NUM_NEURONS; i++){
        #pragma HLS PIPELINE
        for (int j = 0; j < INPUT_SIZE; j++){
            outputData[i*INPUT_SIZE+j] = postAct_prevlayer[j] * dL_dZ[i];
        }
    }
    //paramUpdate<NUM_NEURONS, INPUT_SIZE>(weights, LEARNING_RATE, outputData); // get address for weights from param vector

}

template <int array_sz>
void relu_deriv(            
    fixedP* inputData,
    fixedP* outputData
){
    for (int i = 0; i < array_sz; i++){
        #pragma HLS PIPELINE
        if ((fixedP)inputData[i] < 0){
            outputData[i] = 0;
        }
        else if ((fixedP)inputData[i] > 0){
            outputData[i] = 1;
        }
    }
}

template <int rows_mat1, int cols_mat1, int cols_mat2>
void matMult(
    fixedP* mtxA, 
    fixedP* mtxB, 
    fixedP* mtxC
    ){
    for (int i = 0; i < rows_mat1; i++) {         // Row of A (and C)
    #pragma HLS PIPELINE
        for (int j = 0; j < cols_mat2; j++) {     // Column of B (and C)
            mtxC[i * cols_mat2 + j] = 0;
            for (int k = 0; k < cols_mat1; k++) { // Inner dimension
                mtxC[i * cols_mat2 + j] += mtxA[i * cols_mat1 + k] * mtxB[k * cols_mat2 + j];
            }
        }
    }
}
// for fully connected layers
template <int rows, int cols>
void transpose(fixedP* matrix, fixedP* transposed) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed[j * rows + i] = matrix[i * cols + j];
        }
    }
}



template<int NUM_NEURONS, int INPUT_SIZE, int PREV_LAYER_INPUT_SIZE> // 84, 120, 10
void hiddenLayerBPFC(
    fixedP* inputData,                                      // pre activation output of current layer (generate in pytorch)
    fixedP* postAct_prevlayer,                              // post activation output from the previous layer (layer l-1)
    fixedP* outputData,                                     // resulting data is a matrix of loss gradients, for example [84x10] in current lenet5 implementation 
    fixedP* weights,                                         // weights and biases from layer in front
    fixedP* biases,
    fixedP* frontLayerLoss,                                 // loss computed from previous step (layer l+1)
    fixedP* lossOut,                                   
){
    fixedP dL_dA[NUM_NEURONS];
    fixedP reluDerivs[NUM_NEURONS];                          // int array because this value is entirely 1's and 0's 
    fixedP* reluDerivsFrontLayer;
    fixedP paramsT[NUM_NEURONS*PREV_LAYER_INPUT_SIZE];


    relu_deriv<NUM_NEURONS>(inputData, reluDerivs);
    printf("-----------reluDerivs:----------\n");
    for(int i = 0; i<NUM_NEURONS; i++){ // num_neurons = 84
        printf("%f,",reluDerivs[i].to_float());
    }
    printf("\n");


    // step1: dL/dA = weights matrix(layer ahead) * loss from previous layer (flag == 1)
    // step2: buffer = reluderivs * dl/da
    // step3: outputdata = buffer * activations of previous layer 
    fixedP buffer[10000];
    fixedP bufferTwo[10000];


   
    

    // step1


    transpose<PREV_LAYER_INPUT_SIZE, NUM_NEURONS>(weights, paramsT); //paramsT is now 84x10
    matMult<NUM_NEURONS, PREV_LAYER_INPUT_SIZE, 1 >(paramsT, frontLayerLoss, dL_dA); // [84x10] * [10x1] = 84x1 (RxC)
    printf("-----------dL_dA:----------\n");
    for(int i = 0; i<INPUT_SIZE; i++){ 
        lossOut[i] = dL_dA[i];
        printf("%f,",lossOut[i].to_float());
    }
    printf("\n");
    

    // step2
    
    printf("-----------buffer(reluderivs*dl_da):----------\n");
    for(int i = 0; i < NUM_NEURONS; i++){
        buffer[i]=reluDerivs[i]*dL_dA[i]; // this yields negative values so it cannot be this 
        printf("%f,",buffer[i].to_float()); 
    }
    printf("\n");


    // step3
    matMult<NUM_NEURONS, 1, INPUT_SIZE>(buffer, postAct_prevlayer, outputData); // 84, 1, 120

}

template<int input_depth, int input_rows, int input_cols> // note these are w.r.t. the flatten layer not the bp function
void unflatten_bp(
    fixedP* lossTerm,
    fixedP* output
){
    int idx = 0;
    for (int d = 0; d < input_depth; d++) {   
        for (int i = 0; i < input_rows; i++) { 
            for (int j = 0; j < input_cols; j++) { 
                output[d][i][j] = lossTerm[idx++];
            }
        }
    }
    
}

// new!!--------------------------------
template<int inputDepth, int numKernels, int kernelSize>
void rotateKernels(fixedP* weights) {
    for (int k = 0; k < numKernels; ++k) {
        for (int c = 0; c < inputDepth; ++c) {
            // Pointers to the start of the kernel
            int baseIndex = k * inputDepth * kernelSize + c * kernelSize;
            
            // Flip the kernel in place
            for (int i = 0; i < kernelSize / 2; ++i) {
                fixedP temp = weights[baseIndex + i];
                weights[baseIndex + i] = weights[baseIndex + (kernelSize - 1 - i)];
                weights[baseIndex + (kernelSize - 1 - i)] = temp;
            }
        }
    }
}

template<int inputDepth, int inputCols, int numKernels, int kernelSize, int padding, int layerOutputSize>
void conv1D_bp(
    fixedP* inputData,               // Pre-activation output of current layer (layerOutputSize), output of forward prop
    fixedP* postAct_prevlayer,       // Post-activation output from previous layer (input in PyTorch)
    fixedP* outputData,              // Matrix of loss gradients
    fixedP* weights,                  // Weights (kernels)
    fixedP* biases,
    fixedP* frontLayerLoss,          // Loss gradients from the previous layer
    fixedP* lossOut                  // Intermediate BRAM for storing dl_da (gradient w.r.t. activations)
) {
    // Flattened arrays (1D index access)
    fixedP conv_result[layerOutputSize]; // Store the intermediate convolution result
    fixedP dl_dw[numKernels * inputDepth * kernelSize];  // To store the weight gradients
    fixedP inputGradients[inputDepth*inputCols];
    if (kernelSize > 1){
        rotateKernels<inputDepth, numKernels, kernelSize>(weights); //new!!!!!
    }
    // 1. Loss to prop backwards
    int output_width = inputCols + 2*padding - (kernelSize) + 1; // also divide by stride which is 1 in this case 

    for (int kernelIndex = 0; kernelIndex < numKernels; ++kernelIndex) {
        for (int outputIndex = 0; outputIndex < output_width; ++outputIndex) {
            for (int inputChannel = 0; inputChannel < inputDepth; ++inputChannel) {
                // Adjust the input index based on padding
                int paddedIndex = outputIndex - padding;  // Shift by padding
    
                // Ensure the index stays within the valid bounds (it shouldn't go below 0)
                if (paddedIndex < 0 || paddedIndex >= inputCols) {
                    continue; // Skip invalid indices caused by padding
                }
    
                // Iterate over the kernel size
                for (int kernelElement = 0; kernelElement < kernelSize; ++kernelElement) {
                    // Check the bounds for the kernel access
                    if (paddedIndex + kernelElement < 0 || paddedIndex + kernelElement >= inputCols) {
                        continue; // Skip if the kernel would go out of bounds
                    }
    
                    // Get the kernel value at (kernelIndex, inputChannel, kernelElement)
                    fixedP kernelValue = weights[kernelIndex * inputDepth * kernelSize + inputChannel * kernelSize + kernelElement];
    
                    // Get the loss gradient for this output element
                    fixedP lossGradient = frontLayerLoss[kernelIndex * output_width + outputIndex];
    
                    // Compute the gradient for the input tensor by applying the kernel and the loss gradient
                    int inputIndex = paddedIndex + kernelElement;  // Adjust the input index by kernel element
    
                    // Add the contribution to the input gradient
                    inputGradients[inputChannel * inputCols + inputIndex] += lossGradient * kernelValue;
                }
            }
        }
    }



    // 2. Weight Gradients 
    for (int k = 0; k < numKernels; ++k) {
        for (int c = 0; c < inputDepth; ++c) {
            for (int j = 0; j < kernelSize; ++j) {
                fixedP acc = 0;
                for (int t = 0; t < output_width; ++t) {
                    int input_index = t + j;
                    acc += postAct_prevlayer[c * inputCols + input_index] * frontLayerLoss[k * output_width + t];
                }
                dl_dw[k * inputDepth * kernelSize + c * kernelSize + j] = acc;
            }
        }
    }

    int counter = 0;
    for (int i = 0; i < numKernels*inputDepth*kernelSize; i++ ){
        outputData[i] = dl_dw[i];
        counter++;
    }
    printf("Number of gradients: %i\n", counter);

    // Store dl_da in lossOut for further use
    for (int i = 0; i < layerOutputSize; i++) {
        lossOut[i] = inputGradients[i];
    }
}


//unpooling function has to unpool the gradient to match dimensions of next layer 
template<int INPUT_SIZE, int KERNEL_SIZE, int STRIDE> //input_size is the size of the prepoolInput
void maxunpool1d(
        fixedP* prepoolInput,           // input to the maxpooling layer
        fixedP* pooledOutput,           // output of the maxpooling layer
        fixedP* pooledGrad,             // gradient vector that is to be unpooled (separately pass weight and bias)(input)
        fixedP* unpooledGrad            // unpooledGradient that is input for layer (L-1)

){

    int POOLED_SIZE = ((INPUT_SIZE-KERNEL_SIZE)/ STRIDE + 1);
    // For each pooled output window
    for (int i = 0; i < POOLED_SIZE; ++i) {
        int window_start = i * STRIDE;
        fixedP max_val = -1000; // value must be LOWER than any value we are expecting 
        int max_idx = -1;

        // Recompute the index of the max value within the window
        for (int k = 0; k < KERNEL_SIZE; ++k) {
            int idx = window_start + k;
            if (idx < INPUT_SIZE && input[idx] > max_val) {
                max_val = input[idx];
                max_idx = idx;
            }
        }
        
        // // Double-check that this value matches pooled output (optional but safe)
        // if (max_idx >= 0 && max_idx < INPUT_SIZE && input[max_idx] == pooled_output[i]) {
             unpooledGrad[max_idx] = pooledGrad[i];
        // }
    }

}







    
    
    
    
    
    

// #include "bp_consts.h" 
#include "engine.h"
// #include "model_consts.h"
#include "globals.h"

 
void bp( 
	volatile fixedP* intermediateResultIF,          
	volatile fixedP* groundTruthLabels,
	volatile fixedP* gradients,
    volatile fixedP* weightsBufferIF,
    volatile fixedP* biasBufferIF
) {



#pragma HLS INTERFACE mode=m_axi port=intermediateResultBuffer      bundle=backpropBundle depth=MAX_NUM_WORDS
#pragma HLS INTERFACE mode=m_axi port=groundTruthLabels             bundle=backpropBundle depth=OUTPUT_SIZE
#pragma HLS INTERFACE mode=m_axi port=gradients                     bundle=backpropBundle depth=MAX_NUM_GRADIENTS
#pragma HLS INTERFACE mode=m_axi port=weights                       bundle=backpropBundle depth=MAX_NUM_WEIGHTS         // determine depth here and below 
#pragma HLS INTERFACE mode=m_axi port=biasBufferIF                  bundle=backpropBundle depth=MAX_NUM_BIAS

#include "backprop_impl.h"
}
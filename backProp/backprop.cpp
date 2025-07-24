#include "bp_consts.h" 
#include "engine.h"

 
void top( 
	 fixedP labels[OUTPUT_SIZE],			// one-hot vector stored in ddr? bram? this is v small so would be a shame to store in bram/ddr
	 volatile fixedP* intermediateResultBuffer,
	 volatile fixedP* postActPrevLayer,
	 volatile fixedP* outputGradients,
	 volatile fixedP* weights,
	 volatile fixedP* bias

) {

// #pragma HLS INTERFACE bram 	port=inputData 			storage_type=ram_1p
// #pragma HLS INTERFACE bram 	port=outputData 		storage_type=ram_1p
// #pragma HLS INTERFACE bram  port=paramVector 		storage_type=ram_1p
// #pragma HLS RESOURCE  		variable=inputData 	  	core=RAM_1P_BRAM 			        // Assigns resource type to a given variable, so variable inputData will be implemented using a single port BRAM
// #pragma HLS RESOURCE  		variable=outputData 	core=RAM_1P_BRAM 
// #pragma HLS RESOURCE  		variable=paramVector 	core=RAM_1P_BRAM 
// #pragma HLS bind_storage  	variable=paramVector 	type=ram_1p  impl=bram              // Specifies memory type, and how it will be implemented in RTL, ie what logic resource will be consumed to implement
// #pragma HLS bind_storage  	variable=inputData   	type=ram_1p  impl=bram 
// #pragma HLS bind_storage  	variable=outputData  	type=ram_1p  impl=bram 
// #pragma HLS INTERFACE 		s_axilite port=return   bundle=CTRL_BUS                     /* Just means that we have start and stop signals*/ 

// #pragma HLS INTERFACE mode=m_axi port=intermediateResultBuffer 	bundle=backpropValues depth=MAX_NUM_WORDS // this AXI I/F will grab FP predictions & intermediate resuls 
// #pragma HLS INTERFACE mode=m_axi port=postActPrevLayer 			bundle=backpropValues depth=MAX_NUM_WORDS
// #pragma HLS INTERFACE mode=m_axi port=outputGradients 			bundle=backpropValues depth=MAX_NUM_WORDS
// #pragma HLS INTERFACE mode=m_axi port=weights 					bundle=backpropValues depth=MAX_NUM_WORDS
// #pragma HLS INTERFACE mode=m_axi port=bias 						bundle=backpropValues depth=MAX_NUM_WORDS
#include "backprop_impl.h"
}
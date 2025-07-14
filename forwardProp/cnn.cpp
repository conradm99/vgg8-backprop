#include "conv.h" 
#include "pooling.h" 
#include "FullyConnected.h" 
#include "utils.h" 
#include "model_consts.h" 
void top( 
	 fixedP inputData  [INPUT_SIZE],
	 fixedP outputData [OUTPUT_SIZE],
	 fixedP paramVector[NUM_PARAMETERS] 

) {

#pragma HLS INTERFACE bram 	port=inputData storage_type=ram_1p
#pragma HLS INTERFACE bram 	port=outputData storage_type=ram_1p
#pragma HLS INTERFACE bram  port=paramVector storage_type=ram_1p
#pragma HLS RESOURCE  variable=inputData 	  core=RAM_1P_BRAM 
#pragma HLS RESOURCE  variable=outputData 	  core=RAM_1P_BRAM 
#pragma HLS RESOURCE  variable=paramVector 	  core=RAM_1P_BRAM 
#pragma HLS bind_storage  variable=paramVector type=ram_1p  impl=bram 
#pragma HLS bind_storage  variable=inputData type=ram_1p  impl=bram 
#pragma HLS bind_storage  variable=outputData type=ram_1p  impl=bram 
#pragma HLS INTERFACE s_axilite port=return   bundle=CTRL_BUS /* Just means that we have start and stop signals*/ 

#include "model_impl.h" 
}
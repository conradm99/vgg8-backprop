#include <stdio.h>
#include <cmath> 
#include "params.h" 
#include "input.h"
#include <ap_fixed.h> 
#include "testvals.h"
typedef ap_fixed<32,10> fixedP;

#define INPUTSIZE  1024
#define OUTPUTSIZE  10
#define NUMPARAMETERS  61706
#define NUM_WORDS 15002 // total number of params for that buffer

void top(
	 volatile fixedP*,volatile fixedP*,volatile fixedP*,volatile fixedP*,volatile fixedP*
);

int main() { 

	 volatile fixedP intermediateResultBufferFixed[NUM_WORDS]; //buffer that stores all intermediate results from forward pass
	 volatile fixedP groundTruthLabels[OUTPUTSIZE];
	 volatile fixedP gradientsbus[48000];                       // gradients generated during BP
     volatile fixedP weightsBuffer[];                           // buffer that contains all weights
     volatile fixedP biasBuffer[];                              // contains all biases


	 for(int i = 0; i < OUTPUTSIZE; i++)
	 	groundTruthLabels[i] = 0;

	groundTruthLabels[2] = 1;

		
	//  for(int i = 0; i < NUM_WORDS; i++) 
	// 	 intermediateResultBufferFixed[i] = (fixedP)intermediateResultBuffer[i];

	 top(inputDataFixed, outDataFixed, paramVectorFixed, paramsL5Fixed, paramsL4Fixed, intermediateResultBufferFixed, groundTruthLabels, gradientsbus );

	 for(int i = 0; i < OUTPUTSIZE; i++) { 
		 printf("Seen: %f ", outDataFixed[i].to_float());
	 } 
	 printf("out = ["); 
	 for(int i = 0; i < OUTPUTSIZE; i++) { 
		 printf("%f,", outDataFixed[i].to_float()); 
	 } 
	 printf("];");

 	 return 0; 
 }

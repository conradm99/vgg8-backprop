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
#define NUM_WORDS 15002

void top(
	 fixedP*, fixedP*, fixedP*,fixedP*,fixedP*, volatile fixedP*,volatile fixedP*,volatile fixedP*
);

int main() { 

	 fixedP outDataFixed[OUTPUTSIZE];
	 fixedP inputDataFixed[INPUTSIZE]; 
	 fixedP paramVectorFixed[NUMPARAMETERS];
	 volatile fixedP intermediateResultBufferFixed[NUM_WORDS];
	 volatile fixedP groundTruthLabels[OUTPUTSIZE];
	 volatile fixedP gradientsbus[48000];
	 fixedP paramsL5Fixed[840];
	 fixedP paramsL4Fixed[10080];

	 for(int i = 0; i < OUTPUTSIZE; i++)
	 	groundTruthLabels[i] = 0;

	groundTruthLabels[2] = 1;

	 for(int i = 0; i < INPUTSIZE; i++) 
		 inputDataFixed[i] = (fixedP)inputData[i];

	 for(int i = 0; i < NUMPARAMETERS; i++)
		 paramVectorFixed[i] = (fixedP)paramVector[i];

	 for(int i = 0; i<10080;i++)
		 paramsL4Fixed[i] = weights_l4[i];

	for(int i = 0; i<840;i++)
		 paramsL5Fixed[i] = weights_l5[i];
		
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

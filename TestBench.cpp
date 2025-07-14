#include <stdio.h>
#include <cmath> 
#include "params.h" 
#include "input.h"
#include <ap_fixed.h> 
typedef ap_fixed<32,10> fixedP;

#define INPUTSIZE  2048
#define OUTPUTSIZE  24
#define NUMPARAMETERS  430832

void top(
	 fixedP*, fixedP*, fixedP* 
);

int main() { 

	 fixedP outDataFixed[OUTPUTSIZE];
	 fixedP inputDataFixed[INPUTSIZE]; 
	 fixedP paramVectorFixed[NUMPARAMETERS];

	 for(int i = 0; i < INPUTSIZE; i++) 
		 inputDataFixed[i] = (fixedP)inputData[i];

	 for(int i = 0; i < NUMPARAMETERS; i++)
		 paramVectorFixed[i] = (fixedP)paramVector[i];

	 top(inputDataFixed, outDataFixed , paramVectorFixed);

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

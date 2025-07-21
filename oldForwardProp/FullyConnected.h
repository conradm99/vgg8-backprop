#include "/home/miszczak-c/Documents/Vitis_Libraries/vision/L1/include/common/xf_video_mem.hpp"
//#include <gmp.h> 
#include <hls_stream.h>
#include <hls_math.h>
#include "ap_axi_sdata.h"
#include <ap_fixed.h>

//#define IS_PIPELINED

typedef  ap_fixed<32,10> fixedP;
//typedef  ap_fixed<64,30> doublefixedp; 
//typedef ap_uint<6> uint6; 


enum Activation {
	LINEAR,
	SOFTMAX,
	SIGMOID,
	RELU
};


template<int INPUT_SIZE>
void generateOutputForOneNeuron(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* weights,
		fixedP* bias
		) {

	*outputData = 0;

	for(int idx = 0; idx < INPUT_SIZE; idx++) {

#ifdef IS_PIPELINED
		#pragma HLS PIPELINE
#endif

		*outputData +=  (fixedP)weights[idx] * inputData[idx];

	}

	*outputData += (fixedP)*bias;



}

template<int INPUT_SIZE, int NUM_NEURONS, int ACTIVATION>
void fullyConnectedLayer(
		fixedP* inputData,
		fixedP* outputData,
		fixedP* params
	) {


	for (int idxNeuron = 0; idxNeuron < NUM_NEURONS; idxNeuron++) {

			generateOutputForOneNeuron
			<INPUT_SIZE>
			(
				inputData,
				outputData + idxNeuron, 		 				/* Each neuron produces exactly one output */
				params + idxNeuron * INPUT_SIZE, 				/* Each neuron has INPUT_SIZE weights */
				params +  NUM_NEURONS * INPUT_SIZE + idxNeuron /* and one bias */
			);
	}

}





template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS>
void flattenLayerKeras(
		fixedP* inputData,
		fixedP* outputData
	) {

	int outIdx = 0;

	for (int idxCol = 0; idxCol < INPUT_COLS; idxCol++) {

#ifdef IS_PIPELINED
		#pragma HLS PIPELINE
#endif

		for (int idxRow = 0; idxRow < INPUT_ROWS; idxRow++) {
			for (int idxDepth = 0; idxDepth < INPUT_DEPTH; idxDepth++) {
				outputData[outIdx++] = inputData[idxDepth * INPUT_ROWS * INPUT_COLS + (idxRow * INPUT_COLS) + idxCol];
			}
		}
	}
}



template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS>
void flattenLayer(
		fixedP* inputData,
		fixedP* outputData
	) {

	

	int outIdx = 0;
for (int idxDepth = 0; idxDepth < INPUT_DEPTH; idxDepth++) {
	

#ifdef IS_PIPELINED
		#pragma HLS PIPELINE
#endif

		for (int idxRow = 0; idxRow < INPUT_ROWS; idxRow++) {
			for (int idxCol = 0; idxCol < INPUT_COLS; idxCol++) {
				outputData[outIdx++] = inputData[idxDepth * INPUT_ROWS * INPUT_COLS + (idxRow * INPUT_COLS) + idxCol];
			}
		}
	}
}


template<int INPUT_DEPTH, int INPUT_ROWS, int INPUT_COLS>
void ReLU(
		fixedP* inputData,
		fixedP* outputData,
		fixedP alpha
	) {

	int outputsSoFar = 0;

	for (int idxDepth = 0; idxDepth < INPUT_DEPTH; idxDepth++){


		for (int idx = 0; idx < INPUT_ROWS * INPUT_COLS; idx++) {

#ifdef IS_PIPELINED
		#pragma HLS PIPELINE
#endif


			if(inputData[outputsSoFar] >= 0)
				outputData[outputsSoFar] = inputData[outputsSoFar];
			else
				outputData[outputsSoFar] = inputData[outputsSoFar] * alpha;
			outputsSoFar++;

		}

	}
}




float HLSexponent(fixedP x) {
    // Convert fixed-point to float for the exponential operation
    float xf = static_cast<float>(x);
    // Calculate the exponential
    float exp_xf = hls::exp(xf);

    return exp_xf;
}


/*
uint6 get_msb_fractional(doublefixedp fractional_part) {
    // Shift left to bring the 6 MSBs of the fractional part to the integer part
    uint6 msb_fractional = (fractional_part * (1 << 6)).to_uint();
    return msb_fractional;
}

doublefixedp LUTexponent(fixedP x){
	//The implementation of exponential function following the Data flow Graph in : 
	//A fixed point exponential function accelerator for a neuromorphic many-core system

	//INTEGER PART LUT//
	//20 bits for integer part of output --> MaxInt = 2^30-1 ~ exp(20.79)  & MinInt = 2^-29 ~ exp(-20.1)
	//If input integer part is larger than 20 --> Saturation, Lower than -20 --> Zero
	doublefixedp MaxSaturation = 536870911.99999999976716935634613037109375;
	//Const will help vitis synthetise this as a ROM /
	const doublefixedp IntegerLUT[20 + 20 + 1] = {2.061153622438558e-09, 	5.602796437537268e-09, 	1.522997974471263e-08, 	4.139937718785167e-08, 	1.1253517471925912e-07, 	3.059023205018258e-07, 	8.315287191035679e-07, 	2.2603294069810542e-06, 	6.14421235332821e-06, 	1.670170079024566e-05, 	4.5399929762484854e-05, 	0.00012340980408667956, 	0.00033546262790251185, 	0.0009118819655545162, 	0.0024787521766663585, 	0.006737946999085467, 	0.01831563888873418, 	0.049787068367863944, 	0.1353352832366127, 	0.36787944117144233, 	1.0, 	2.718281828459045, 	7.38905609893065, 	20.085536923187668, 	54.598150033144236, 	148.4131591025766, 	403.4287934927351, 	1096.6331584284585, 	2980.9579870417283, 	8103.083927575384, 	22026.465794806718, 	59874.14171519782, 	162754.79141900392, 	442413.3920089205, 	1202604.2841647768, 	3269017.3724721107, 	8886110.520507872, 	24154952.7535753, 	65659969.13733051, 	178482300.96318725, 	485165195.4097903 };
	

	int IntPart = (int) x;
	if (IntPart > 20){
		return MaxSaturation; //Saturation to Max Value
	}
	else{
		if (IntPart < -20 )
			return 0.0;
	}
	#pragma HLS pipeline
	int IntLUTaddr = IntPart + 20;
	doublefixedp IntContribution = IntegerLUT[IntLUTaddr];

	//Fractional Part LUT --> helps reduce the error introduced by Taylor approximation/
	//6LSBs --> Access LUT, 16LSBs --> Taylor Approx //
	fixedP fractional_part = x - IntPart;
	uint6  msbs_fract = get_msb_fractional(fractional_part);
	const   doublefixedp MSBsFractLUT[32] =  { doublefixedp(1.0), 	 doublefixedp(1.0317434074991028), doublefixedp(1.0644944589178593), doublefixedp(1.0982851403078258), doublefixedp(1.1331484530668263), doublefixedp(1.1691184461695043), doublefixedp(1.2062302494209807), doublefixedp(1.2445201077660952), doublefixedp(1.2840254166877414), doublefixedp(1.3247847587288655), doublefixedp(1.3668379411737963), doublefixedp(1.4102260349257107), doublefixedp(1.4549914146182013), doublefixedp(1.5011778000001228), doublefixedp(1.5488302986341331), doublefixedp(1.5979954499506333), doublefixedp(1.6487212707001282), doublefixedp(1.7010573018484008), doublefixedp(1.7550546569602985), doublefixedp(1.8107660721193872), doublefixedp(1.8682459574322223), doublefixedp(1.9275504501675447), doublefixedp(1.988737469582292), doublefixedp(2.0518667734879767), doublefixedp(2.117000016612675), doublefixedp(2.184200810815618), doublefixedp(2.2535347872132085), doublefixedp(2.325069660277121), doublefixedp(2.398875293967098), doublefixedp(2.475023769963025), doublefixedp(2.553589458062927), doublefixedp(2.6346490888156313) 	}; 
	const   doublefixedp ToRemove[32] =  {0, 	0.03125, 	0.0625, 	0.09375, 	0.125, 	0.15625, 	0.1875, 	0.21875, 	0.25, 	0.28125, 	0.3125, 	0.34375, 	0.375, 	0.40625, 	0.4375, 	0.46875, 	0.5, 	0.53125, 	0.5625, 	0.59375, 	0.625, 	0.65625, 	0.6875, 	0.71875, 	0.75, 	0.78125, 	0.8125, 	0.84375, 	0.875, 	0.90625, 	0.9375, 	0.96875 	}; 
	doublefixedp MSBsFractContribution = MSBsFractLUT[msbs_fract];

	//Taylor with the LSBs/
	const doublefixedp  c2 =  0.5 ;
	const doublefixedp c3 =  0.125 + 0.03125 + 0.0078125 + 0.001953125 + 0.00048828125 + 0.0001220703125; 
	const doublefixedp  c4 =  0.03125 + 0.0078125 + 0.00390625 ;

	doublefixedp msb_removed_fractional_part = fractional_part - ToRemove[msbs_fract];
	doublefixedp Tylor1order = msb_removed_fractional_part *MSBsFractContribution ;
	doublefixedp Tylor2order = msb_removed_fractional_part * msb_removed_fractional_part * MSBsFractContribution;
	doublefixedp Tylor3order = msb_removed_fractional_part * msb_removed_fractional_part * msb_removed_fractional_part * MSBsFractContribution;
	doublefixedp Tylor4order = msb_removed_fractional_part * msb_removed_fractional_part * msb_removed_fractional_part * MSBsFractContribution;

	//The fractional part contribution is a combination of the Tylor expansion and LUT /
	doublefixedp FractionalPartContribution = MSBsFractContribution + 
											  Tylor1order + 
											  Tylor2order*c2 + 
											  Tylor3order*c3 + 
											  Tylor4order*c4; 
	//Adding the contribution of Integer Part and returning/
	doublefixedp Result = IntContribution * FractionalPartContribution;
	return Result;


}

*/

/*
template<int INPUT_NUERONS>
void SoftMax(
		fixedP* inputData,
		fixedP* outputData
	) {

		doublefixedp Buffer[INPUT_NUERONS];
		doublefixedp Sum = 0.0;
		for(int n = 0 ; n < INPUT_NUERONS; n++){
			#pragma HLS PIPELINE
			Buffer[n] = LUTexponent(inputData[n]);
			Sum += Buffer[n];
		}
		for(int n = 0; n < INPUT_NUERONS; n++){
			#pragma HLS pipeline
			outputData[n] = (fixedP) Buffer[n] / Sum;
		}

}
*/

template<int INPUT_NUERONS>
void SoftMax(
		fixedP* inputData,
		float* outputData
	) {

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
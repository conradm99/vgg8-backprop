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

const char classes[24][16] = {
    "32PSK", "16APSK", "32QAM", "FM", "GMSK", "32APSK",
    "OQPSK", "8ASK", "BPSK", "8PSK", "AM-SSB-SC", "4ASK",
    "16PSK", "64APSK", "128QAM", "128APSK", "AM-DSB-SC",
    "AM-SSB-WC", "64QAM", "QPSK", "256QAM", "AM-DSB-WC",
    "OOK", "16QAM"
};

// HLS-friendly string matching to get hyperclass
const char* get_hyperclass(const char* mod) {
    if (!strcmp(mod, "4ASK") || !strcmp(mod, "8ASK") || !strcmp(mod, "OOK"))
        return "ASK";
    if (!strcmp(mod, "BPSK") || !strcmp(mod, "8PSK") || !strcmp(mod, "16PSK") ||
        !strcmp(mod, "32PSK") || !strcmp(mod, "OQPSK") || !strcmp(mod, "QPSK"))
        return "PSK";
    if (!strcmp(mod, "16APSK") || !strcmp(mod, "32APSK") ||
        !strcmp(mod, "64APSK") || !strcmp(mod, "128APSK"))
        return "APSK";
    if (!strcmp(mod, "16QAM") || !strcmp(mod, "32QAM") || !strcmp(mod, "64QAM") ||
        !strcmp(mod, "128QAM") || !strcmp(mod, "256QAM"))
        return "QAM";
    if (!strcmp(mod, "FM") || !strcmp(mod, "GMSK"))
        return "fM";
    if (!strcmp(mod, "AM-DSB-SC") || !strcmp(mod, "AM-SSB-WC") ||
        !strcmp(mod, "AM-SSB-SC") || !strcmp(mod, "AM-DSB-WC"))
        return "aM";
    return "Unknown";
}

// Core logic: find max index and lookup modulation class
void classify_output(const fixedP logits[24]) {
    int max_index = 0;
    fixedP max_val = logits[0];

    // Simple max finder loop (HLS-friendly)
    for (int i = 1; i < 24; ++i) {
        if ((fixedP)logits[i] > (fixedP)max_val) {
            max_val = logits[i];
            max_index = i;
        }
    }

    // Lookup modulation and hyperclass
    const char* modulation = classes[max_index];
    const char* hyper = get_hyperclass(modulation);

    // Print results (replace with AXI stream or hardware output if needed)
    std::cout << "Max index: " << max_index << "\n";
    std::cout << "Modulation: " << modulation << "\n";
    std::cout << "Hyperclass: " << hyper << "\n";
}

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

	 classify_output(outDataFixed);

 	 return 0; 
 }

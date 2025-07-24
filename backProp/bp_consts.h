// Layer FULLYCONNECTED2
#define FULLYCONNECTED2_IN_FEATURES 100
#define FULLYCONNECTED2_OUT_FEATURES 24

// Layer FULLYCONNECTED1
#define FULLYCONNECTED1_IN_FEATURES 100
#define FULLYCONNECTED1_OUT_FEATURES 100

#define FULLYCONNECTED1_PREV_LAYER_INPUT_SIZE 24

// Layer FULLYCONNECTED0
#define FULLYCONNECTED0_IN_FEATURES 4096
#define FULLYCONNECTED0_OUT_FEATURES 100

#define FULLYCONNECTED0_PREV_LAYER_INPUT_SIZE 100

#define MAX_INPUTDATA_SIZE 32768
static fixedP inputData_ping[MAX_INPUTDATA_SIZE];
static fixedP inputData_pong[MAX_INPUTDATA_SIZE];

#define MAX_OUTPUTDATA_SIZE 32768
static fixedP outputData_ping[MAX_OUTPUTDATA_SIZE];
static fixedP outputData_pong[MAX_OUTPUTDATA_SIZE];

#define MAX_POSTACT_PREV_SIZE 32768
static fixedP postAct_prev_ping[MAX_POSTACT_PREV_SIZE];
static fixedP postAct_prev_pong[MAX_POSTACT_PREV_SIZE];

#define MAX_WEIGHTS_SIZE 409600
static fixedP weights_ping[MAX_WEIGHTS_SIZE];
static fixedP weights_pong[MAX_WEIGHTS_SIZE];

#define MAX_BIASES_SIZE 100
static fixedP biases_ping[MAX_BIASES_SIZE];
static fixedP biases_pong[MAX_BIASES_SIZE];

#define MAX_FRONTLAYERLOSS_SIZE 32768
static fixedP frontLayerLoss_ping[MAX_FRONTLAYERLOSS_SIZE];
static fixedP frontLayerLoss_pong[MAX_FRONTLAYERLOSS_SIZE];

#define MAX_LOSSOUT_SIZE 32768
static fixedP lossOut_ping[MAX_LOSSOUT_SIZE];
static fixedP lossOut_pong[MAX_LOSSOUT_SIZE];

static fixedP weightGrads[MAX_WEIGHTS_SIZE]; // buffer to store gradients for weights and biases
static fixedP weightsBuffer[NUM_WEIGHTS_0+NUM_WEIGHTS_1+NUM_WEIGHTS_2]; // buffer to store weights that will be updated


const int weights_DDR_OFFSET0 = 0; 
const int weights_DDR_OFFSET1 = FULLYCONNECTED0_OUT_FEATURES * FULLYCONNECTED0_IN_FEATURES;
const int weights_DDR_OFFSET2 = weights_DDR_OFFSET1 + FULLYCONNECTED1_OUT_FEATURES * FULLYCONNECTED1_IN_FEATURES;

const int bias_DDR_OFFSET0 = FULLYCONNECTED0_OUT_FEATURES;
const int bias_DDR_OFFSET1 = bias_DDR_OFFSET0 + FULLYCONNECTED1_OUT_FEATURES;
const int bias_DDR_OFFSET2 = bias_DDR_OFFSET1 + FULLYCONNECTED2_OUT_FEATURES;

const int gradients_DDR_OFFSET0 = 0;
const int gradients_DDR_OFFSET1 = NUM_GRADIENTS_0;
const int gradients_DDR_OFFSET2 = gradients_DDR_OFFSET1 + NUM_GRADIENTS_1;

// Number of weights in a FC layer = num of neurons * num of inputs
const int NUM_WEIGHTS_0 = 409600;
const int NUM_WEIGHTS_1 = 10000;
const int NUM_WEIGHTS_2 = 2400;

// Number of biases in a FC layer = num of neurons
const int NUM_BIASES_0  = 100;
const int NUM_BIASES_1  = 100;
const int NUM_BIASES_2  = 24;

const int NUM_GRADIENTS_0 = NUM_WEIGHTS_0 + NUM_BIASES_0; // 409600 + 100 = 409700
const int NUM_GRADIENTS_1 = NUM_WEIGHTS_1 + NUM_BIASES_1; // 10000 + 100 = 10100
const int NUM_GRADIENTS_2 = NUM_WEIGHTS_2 + NUM_BIASES_2; // 2400 + 24 = 2424
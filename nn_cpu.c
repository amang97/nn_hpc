/* Varun lalwani & Aman Gupta
 ENG EC 527 - Multicore Programming
 Project - Optimized Neural Network with Relu activation
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

///////////////////////////////////////////////////////////////////////////////
// Defination
#define GIG 3.3e9
#define THREADS 1
static const int numInputs = 2;
static const int numHiddenNodes = 2;
static const int numOutputs = 1;
static const int numTrainingSets = 4;
///////////////////////////////////////////////////////////////////////////////
// Calculates Relu(x)
float relu(float x)
{
  if(x < 0)
  {
    return 0;
  }
  else
  {
    return x;
  }
}

// Calculates Derivative of Relu(x)
float drelu(float x)
{
  if(x < 0)
  {
    return 0;
  }
  else if(x == 0)
  {
    return ((float)rand() / (float)RAND_MAX);
  }
}
///////////////////////////////////////////////////////////////////////////////
// Initializer for the weights and the biases for the Neural Network
float init_weight()
{
  return ((float)rand())/((float)RAND_MAX);
}
///////////////////////////////////////////////////////////////////////////////

// Main Function
int main(int argc, char *argv[])
{
  srand(time(0));
  double hiddenLayer[numHiddenNodes];
  double outputLayer[numOutputs];
  double hiddenLayerBias[numHiddenNodes];
  double outputLayerBias[numOutputs];
  double hiddenWeights[numInputs][numHiddenNodes];
  double outputWeights[numHiddenNodes][numOutputs];
  double training_inputs[numTrainingSets][numInputs];
  double training_outputs[numTrainingSets][numOutputs];
  printf("Hello World\n");

  return 0;
}

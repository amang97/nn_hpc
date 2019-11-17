/* Varun lalwani & Aman Gupta
 ENG EC 527 - Multicore Programming
 Project - Optimized Neural Network with Relu activation
*/

// gcc nn_cpu.c -o nn

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

///////////////////////////////////////////////////////////////////////////////
// Defination
#define GIG 3.3e9
#define THREADS 1
#define numInputs = 2;
#define numHiddenLayers = 2;
#define numOutputs = 1;
//static const int numTrainingSets = 4;
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
// Calculates Result(x)
int result_calculator(float * out_array, int length)
{
  int max_value_output_node = 0;
  int i;
  for(i = 1; i < length; i++)
  {
    if(out_array[max_value_output_node] < out_array[i])
    {
      max_value_output_node = i;
    }
  }
  return max_value_output_node;
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
  if(argc == 1)
  {
    printf("Please pass %d numbers as the number of hidden nodes in a hidden layer\n",numHiddenLayers);
  }
  else if(argc != (numHiddenLayers + 1))
  {
    printf("Insufficient quantity of number of hidden nodes provided. Please pass %d numbers\n", numHiddenLayers);
  }
  else
  {
    int i;

    srand(time(0));

    int hidden_Layer_node_count[numHiddenLayers];
    float * hiddenLayer_output[numHiddenLayers];
    float outputLayer_output[numOutputs];
    float hiddenLayerBiases[numHiddenLayers];
    float outputLayerBias[numOutputs];
    float ** hiddenWeights[numHiddenLayers];
    float ** outputWeights;

    for(i = 0; i < numHiddenLayers; i++)
    {
      hidden_Layer_node_count[i] = stoi(argv[i+1]);
      printf("Hidden Layer %d count: %d\n", i, hidden_Layer_node_count[i]);
    }
  }
  /*double hiddenLayer[numHiddenNodes];
  double hiddenWeights[numInputs][numHiddenNodes];
  double outputWeights[numHiddenNodes][numOutputs];
  double training_inputs[numTrainingSets][numInputs];
  double training_outputs[numTrainingSets][numOutputs];*/
  printf("Hello World\n");

  return 0;
}

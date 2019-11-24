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
#define numInputs 784
#define numHiddenLayers 2
#define numOutputs 2
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
    printf("Please pass %d numbers in the argument\n",numHiddenLayers);
  }
  else if(argc != (numHiddenLayers + 1))
  {
    printf("Insufficient number of arguments provided. Please pass %d numbers\n", numHiddenLayers);
  }
  else
  {
    int i;

    srand(1527);

    int hidden_Layer_node_count[numHiddenLayers];
    float * hiddenLayer_output[numHiddenLayers];
    float outputLayer_output[numOutputs];
    float hiddenLayerBiases[numHiddenLayers];
    float outputLayerBias[numOutputs];
    // For weights, the row number is the output node number
    // The column number if the input node number
    // so it is in the format of (Ax = b)
    // here x is the values in the input layer
    // here y is the result in the output layer
    float * hiddenWeights[numHiddenLayers];
    float * outputWeights;

    for(i = 0; i < numHiddenLayers; i++)
    {
      hidden_Layer_node_count[i] = atoi(argv[i+1]);
      printf("Hidden Layer %d count: %d\n", i, hidden_Layer_node_count[i]);
    }

    //Initialize hidden weigts
    for(i = 0; i < numHiddenLayers; i++)
    {
      int j;
      if(i == 0)
      {
        hiddenWeights[0] = (float*)calloc(hidden_Layer_node_count[0]*784, sizeof(float));
        if(hiddenWeights[0] == NULL)
        {
          printf("Memory not allocated\n");
          return 0;
        }
        for(j = 0 ; j < hidden_Layer_node_count[0]*784; j++)
        {
          hiddenWeights[0][j] = init_weight();
        }
      }
      else
      {
        hiddenWeights[i] = (float*)calloc(hidden_Layer_node_count[i]*hidden_Layer_node_count[i-1],sizeof(float));
        if(hiddenWeights[i] == NULL)
        {
          printf("Memory not allocated\n");
          return 0;
        }
        for(j = 0 ; j < (hidden_Layer_node_count[i]*hidden_Layer_node_count[i-1]); j++)
        {
          hiddenWeights[i][j] = init_weight();
        }
      }
    }

    // Initialize output weights
    outputWeights = (float*)calloc(numOutputs*hidden_Layer_node_count[numHiddenLayers-1], sizeof(float));
    if(outputWeights == NULL)
    {
      printf("Memory not allocated\n");
      return 0;
    }
    for(i = 0; i < (numOutputs*hidden_Layer_node_count[numHiddenLayers-1]); i++)
    {
      outputWeights[i] = init_weight();
    }
    printf("Initialized Weights\n");

    for(i = 0; i < numHiddenLayers; i++)
    {
      free(hiddenWeights[i]);
    }
    free(outputWeights);

    printf("Freed Weights\n");
  }
  /*double hiddenLayer[numHiddenNodes];
  double hiddenWeights[numInputs][numHiddenNodes];
  double outputWeights[numHiddenNodes][numOutputs];
  double training_inputs[numTrainingSets][numInputs];
  double training_outputs[numTrainingSets][numOutputs];*/
  printf("Hello World\n");

  return 0;
}

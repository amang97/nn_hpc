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
#define NUMINPUTS 2
#define NUMHIDDENLAYERS 1
#define NUMOUTPUTS 1
#define BATCH_SIZE 4
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
  else
  {
    return 1;
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
// Matrix Vector multiplication
void mvmr(float* A, float * x, float * result, int num_row, int num_col, int bias)
{
  int i,j;
  for(i = 0; i < num_row; i++)
  {
    float sum = 0;
    for(j = 0; j < num_col; j++)
    {
      sum = sum + (A[(i*num_col) + j] * x[j]);
    }
    result[i] = relu(sum + bias);
  }

}
///////////////////////////////////////////////////////////////////////////////

// Main Function
int main(int argc, char *argv[])
{
  if(argc == 1)
  {
    printf("Please pass %d numbers in the argument\n",NUMHIDDENLAYERS);
  }
  else if(argc != (NUMHIDDENLAYERS + 1))
  {
    printf("Insufficient number of arguments provided. Please pass %d numbers\n", NUMHIDDENLAYERS);
  }
  else
  {
    int i;

    srand(1527);

    int hidden_Layer_node_count[NUMHIDDENLAYERS];
    float training_inputs[NUMINPUTS];
    float * hiddenLayer_output[NUMHIDDENLAYERS];
    float outputLayer_output[NUMOUTPUTS];
    float hiddenLayerBiases[NUMHIDDENLAYERS];
    float outputLayerBias;
    // For weights, the row number is the output node number
    // The column number if the input node number
    // so it is in the format of (Ax = b)
    // here x is the values in the input layer
    // here y is the result in the output layer
    float * hiddenWeights[NUMHIDDENLAYERS];
    float * outputWeights;

    // Get Number of nodes in each Hidden Layer ///////////////////////////////
    for(i = 0; i < NUMHIDDENLAYERS; i++)
    {
      hidden_Layer_node_count[i] = atoi(argv[i+1]);
      printf("Hidden Layer %d count: %d\n", i, hidden_Layer_node_count[i]);
    }

    //Initialize hidden weigts ////////////////////////////////////////////////
    for(i = 0; i < NUMHIDDENLAYERS; i++)
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

    // Initialize output weights////////////////////////////////////////////////
    outputWeights = (float*)calloc(NUMOUTPUTS*hidden_Layer_node_count[NUMHIDDENLAYERS-1], sizeof(float));
    if(outputWeights == NULL)
    {
      printf("Memory not allocated\n");
      return 0;
    }
    for(i = 0; i < (NUMOUTPUTS*hidden_Layer_node_count[NUMHIDDENLAYERS-1]); i++)
    {
      outputWeights[i] = init_weight();
    }
    printf("Initialized Weights for Hidden and Output Layer\n");

    // Initialize biases //////////////////////////////////////////////////////
    // For Hidden Layers
    for(i = 0; i < NUMHIDDENLAYERS; i++)
    {
      hiddenLayerBiases[i] = init_weight();
    }
    // For Output Layer
    outputLayerBias = init_weight();
    printf("Initialized Bias for Hidden and Output Layer\n");

    // Allocate memory to Hidden Layer Output /////////////////////////////////
    for(i = 0; i < NUMHIDDENLAYERS; i++)
    {
      hiddenLayer_output[i] = (float*)calloc(hidden_Layer_node_count[i],sizeof(float));
    }
    printf("Alllocated memory for Hidden Layer Nodes\n");

    // Feed Forward Algorithm /////////////////////////////////////////////////
    for(i = 0; i < NUMHIDDENLAYERS; i++)
    {
      if(i == 0)
      {
        //hiddenLayer_output[i] = mvmr()
      }
    }

    // Free Initialized Weights/////////////////////////////////////////////////
    for(i = 0; i < NUMHIDDENLAYERS; i++)
    {
      free(hiddenWeights[i]);
    }
    free(outputWeights);

    printf("Freed Weights\n");
  }
  printf("Hello World\n");

  return 0;
}

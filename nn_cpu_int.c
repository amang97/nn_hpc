/* Varun lalwani & Aman Gupta
 ENG EC 527 - Multicore Programming
 Project - Optimized Neural Network with Relu activation
*/

// gcc -mavx -lm nn_cpu_int.c -lrt -o nn_int

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <assert.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

///////////////////////////////////////////////////////////////////////////////
// Defination
#define GIG             1000000000
#define THREADS         1
#define INPUTSIZE       784
#define NUMHIDDENLAYERS 1
#define NUMOUTPUTS      10
#define BATCH_SIZE      60
#define EPOCHS          1
#define NUMINPUTSTRAIN  60000
#define NUMINPUTSTEST   10000
#define BUFFER_SIZE     5120
#define LEARN_RATE      0.0001
//static const int numTrainingSets = 4;

///////////////////////////////////////////////////////////////////////////////
// Initializer for the weights and the biases for the Neural Network
float init_weight()
{
  return (((float)rand())/((float)RAND_MAX));
}

///////////////////////////////////////////////////////////////////////////////
// Calculates Relu(x)
float relu(float x)
{
  if(x < 0)
  {
    return (0.01 * x);
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
    return (0.01);
  }
  else
  {
    return 1;
  }
}

// Calculates Sigmoid(x)
float Sigmoid(float x)
{
  return (1 / (1 + exp(-x)));
}

// Calculates Derivative of Sigmoid(x)
float dSigmoid(float x)
{
  return (x * (1 - x));
}
///////////////////////////////////////////////////////////////////////////////
// Calculates Result(x)
int result_calculator(float * out_array, int length)
{
  int max_value_output_node = 0;
  int i;
  float * array_ptr;
  array_ptr = out_array;
  array_ptr++;
  for(i = 1; i < length; i++)
  {
    if(out_array[max_value_output_node] < *(array_ptr))
    {
      max_value_output_node = i;
    }
    array_ptr++;
  }
  return max_value_output_node;
}

// Calculates Time Spent
struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

///////////////////////////////////////////////////////////////////////////////
// Matrix Vector multiplication
void mvmr(float* A, float * x, float * result, int num_row, int num_col, float * bias)
{
  int i,j,k;
  int nLoop = num_col / 4;
  __m128   m1,m2;
  __m128   m0_5 = _mm_set_ps1(0.5f);
  float storer[4];
  __m128*  pDest = (__m128*) storer;
  __m128*  pSrc1 = (__m128*) A;
  float * result_ptr;
  result_ptr = result;
  for(i = 0; i < num_row; i++)
  {
    float sum = 0;

    __m128*  pSrc2 = (__m128*)x;
    __m128*  pSrc3 = (__m128*)bias;
    for(j = 0; j < nLoop; j++)
    {
      //printf("%0.2f,",A[(i*num_col) + j]);
      m1 = _mm_mul_ps(*pSrc1,*pSrc2);
      m2 = _mm_add_ps(m1,*pSrc3);
      *pDest = _mm_add_ps(m2,m0_5);
      pSrc1++;
      pSrc2++;
      pSrc3++;
      sum = sum + storer[0] + storer[1] + storer[2] + storer[3];
    }
    //printf("Here\n");
    *result_ptr = relu(sum);
    result_ptr++;
    //printf("\n%f,",sum);
    //result[i] = sum;
  }
  //printf("\nresult:\n");
  //for(i = 0; i < num_col; i++)
  //{
  //  printf("%f,",x[i]);
  //}
  //printf("\n");
}

void mvms(float* A, float * x, float * result, int num_row, int num_col, float * bias)
{
  int i,j;
  for(i = 0; i < num_row; i++)
  {
    float sum = 0;
    for(j = 0; j < num_col; j++)
    {
      sum = sum + (A[(i*num_col) + j] * x[j]) + bias[i];
    }
    result[i] = Sigmoid(sum);
  }
}

void mvm(float* A, float * x, float * result, int num_row, int num_col)
{
  int i,j;
  for(i = 0; i < num_row; i++)
  {
    float sum = 0;
    for(j = 0; j < num_col; j++)
    {
      sum = sum + (A[(i*num_col) + j] * x[j]);
    }
    result[i] = sum;
  }
}

// Transpose Matrix Vector Multiplication
// b * Transpose(A) = c
void mvmt(float * A, float * b, float * result, int num_row, int num_col)
{
  int i,j;
  for(j = 0; j < num_col; j++)
  {
    float sum = 0;
    for(i = 0; i < num_row; i++)
    {
      sum = sum + (A[(i*num_col) + j] * b[i]);
    }
    result[j] = sum;
  }
}

// a * b = c
void outer_p(float * a, float * b, float * result, int num_row, int num_col)
{
  int i,j;
  int nLoop = num_col / 4;
  __m128   m1;
  __m128   m0_5 = _mm_set_ps1(0.5f);
  __m128*  pDest = (__m128*) result;
  for(i = 0; i < num_row; i++)
  {
    __m128   ma_i = _mm_set_ps1(a[i]);
    __m128*  pSrc2 = (__m128*)b;
    for(j = 0; j < nLoop; j++)
    {
      m1 = _mm_mul_ps(ma_i,*pSrc2);
      *pDest = _mm_add_ps(m1,m0_5);
      pSrc2++;
      pDest++;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Array-Array Operation
void add_weights_l(float * a, float * b, int arr_len)
{
  // optimize indexing
  int i;
  for(i = 0; i < arr_len; i++)
  {
    a[i] += (b[i] * LEARN_RATE / 1000);
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
    int i,j,k,iteration;
    bool store_max = true;
    float what_is_max = 0.0;

    srand(1527);

    int hidden_Layer_node_count[NUMHIDDENLAYERS];
    float training_inputs[NUMINPUTSTRAIN][INPUTSIZE];
    int training_labels[NUMINPUTSTRAIN];
    float testing_inputs[NUMINPUTSTEST][INPUTSIZE];
    int testing_labels[NUMINPUTSTEST];
    float * hiddenLayer_output[NUMHIDDENLAYERS];
    float * hiddenLayerBiases[NUMHIDDENLAYERS];
    float * hiddenLayer_gradient[NUMHIDDENLAYERS];
    float outputLayer_output[NUMOUTPUTS];
    float outputLayerBias[NUMOUTPUTS];
    float outputLayer_gradient[NUMOUTPUTS];
    // For weights, the row number is the output node number
    // The column number if the input node number
    // so it is in the format of (Ax = b)
    // here x is the values in the input layer
    // here y is the result in the output layer
    float * hiddenWeights[NUMHIDDENLAYERS];
    float * hiddenWeights_gradient[NUMHIDDENLAYERS];
    float * outputWeights;
    float * outputWeights_gradient;

    // Get Number of nodes in each Hidden Layer ///////////////////////////////
    for(i = 0; i < NUMHIDDENLAYERS; i++)
    {
      hidden_Layer_node_count[i] = atoi(argv[i+1]);
      printf("Hidden Layer %d count: %d\n", i, hidden_Layer_node_count[i]);
    }

    //Initialize hidden weigts ////////////////////////////////////////////////
    for(i = 0; i < NUMHIDDENLAYERS; i++)
    {
      if(i == 0)
      {
        hiddenWeights[0] = (float*)calloc(hidden_Layer_node_count[0]*INPUTSIZE, sizeof(float));
        hiddenWeights_gradient[0] = (float*)calloc(hidden_Layer_node_count[0]*INPUTSIZE, sizeof(float));
        if((hiddenWeights[0] == NULL) || (hiddenWeights_gradient[0] == NULL))
        {
          printf("Memory not allocated\n");
          return 0;
        }
        for(j = 0 ; j < hidden_Layer_node_count[0]*INPUTSIZE; j++)
        {
          hiddenWeights[0][j] = init_weight();
        }
      }
      else
      {
        hiddenWeights[i] = (float*)calloc(hidden_Layer_node_count[i]*hidden_Layer_node_count[i-1],sizeof(float));
        hiddenWeights_gradient[i] = (float*)calloc(hidden_Layer_node_count[i]*hidden_Layer_node_count[i-1],sizeof(float));
        if((hiddenWeights[i] == NULL) || (hiddenWeights_gradient[i] == NULL))
        {
          printf("Memory not allocated\n");
          return 0;
        }
        for(j = 0 ; j < (hidden_Layer_node_count[i]*hidden_Layer_node_count[i-1]); j++)
        {
          hiddenWeights[i][j] = init_weight();
        }
      }
      hiddenLayerBiases[i] = (float*)calloc(hidden_Layer_node_count[i],sizeof(float));
      if(hiddenLayerBiases[i] == NULL)
      {
        printf("Memory not allocated\n");
        return 0;
      }
    }

    // Initialize output weights////////////////////////////////////////////////
    outputWeights = (float*)calloc(NUMOUTPUTS*hidden_Layer_node_count[NUMHIDDENLAYERS-1], sizeof(float));
    outputWeights_gradient = (float*)calloc(NUMOUTPUTS*hidden_Layer_node_count[NUMHIDDENLAYERS-1], sizeof(float));
    if((outputWeights == NULL) || (outputWeights_gradient == NULL))
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
      for(j = 0; j < hidden_Layer_node_count[i]; j++)
      {
        hiddenLayerBiases[i][j] = init_weight();
      }
    }
    // For Output Layer
    for(i = 0; i < NUMOUTPUTS; i++)
    {
      outputLayerBias[i] = init_weight();
    }
    printf("Initialized Bias for Hidden and Output Layer\n");

    // Allocate memory to Hidden Layer Output /////////////////////////////////
    for(i = 0; i < NUMHIDDENLAYERS; i++)
    {
      hiddenLayer_output[i] = (float*)calloc(hidden_Layer_node_count[i],sizeof(float));
      hiddenLayer_gradient[i] = (float*)calloc(hidden_Layer_node_count[i],sizeof(float));
      if((hiddenLayer_output[i] == NULL) || (hiddenLayer_gradient[i] == NULL))
      {
        printf("Memory not allocated\n");
        return 0;
      }
    }
    printf("Allocated memory for Neural Network Nodes\n");

    // Load Input /////////////////////////////////////////////////////////////
    FILE * pInput;
    char buf[BUFFER_SIZE];
    pInput = fopen("./mnist_train.csv","r");
    if(pInput == NULL)
    {
      printf("Error: Can\'t read or find the training file\n");
    }
    else
    {
      fgets(buf, sizeof(buf), pInput);
      for(iteration = 0 ; iteration < NUMINPUTSTRAIN; iteration++)
      {
        fgets(buf, sizeof(buf), pInput);
        char * tok = strtok(buf, ",");
        training_labels[iteration] = atoi(tok);
        for(i = 0; i < INPUTSIZE; i++)
        {
          tok = strtok(NULL,",");
          if(tok == NULL)
          {
            printf("Can't read Input at %d row, %d column\n",iteration,(i+1));
            break;
          }
          float value = (float)atof(tok);
          if(value != 0)
          {
            value = value / 255;
          }
          training_inputs[iteration][i] = value;
        }
      }
    }
    int closed = fclose(pInput);
    pInput = fopen("./mnist_test.csv","r");
    if(pInput == NULL)
    {
      printf("Error: Can\'t read or find the training file\n");
    }
    else
    {
      fgets(buf, sizeof(buf), pInput);
      for(iteration = 0 ; iteration < NUMINPUTSTEST; iteration++)
      {
        fgets(buf, sizeof(buf), pInput);
        char * tok = strtok(buf, ",");
        testing_labels[iteration] = atoi(tok);
        for(i = 0; i < INPUTSIZE; i++)
        {
          tok = strtok(NULL,",");
          if(tok == NULL)
          {
            printf("Can't read Input at %d row, %d column\n",iteration,(i+1));
            break;
          }
          float value = (float)atof(tok);
          if(value != 0)
          {
            value = value / 255;
          }
          testing_inputs[iteration][i] = value;
        }
      }
    }

    printf("Loaded Training and Testing Inputs\n");

    // Training Algorithm /////////////////////////////////////////////////////

    for(iteration = 0; iteration < EPOCHS; iteration++)
    {
      // Feed Forward Algorithm ///////////////////////////////////////////////
      int correct_answers = 0;
      struct timespec time1, time2;
      struct timespec time_stamp;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
      for(i = 0; i < NUMINPUTSTRAIN; i++)
      {
        for(j = 0; j < NUMHIDDENLAYERS; j++)
        {
          if(j == 0)
          {
            mvmr(hiddenWeights[0], training_inputs[i], hiddenLayer_output[0],hidden_Layer_node_count[0],INPUTSIZE,hiddenLayerBiases[0]);
          }
          else
          {
            mvmr(hiddenWeights[j], hiddenLayer_output[j-1], hiddenLayer_output[j],hidden_Layer_node_count[j],hidden_Layer_node_count[j-1],hiddenLayerBiases[j]);
          }
        }

        mvmr(outputWeights,hiddenLayer_output[NUMHIDDENLAYERS-1],outputLayer_output,NUMOUTPUTS,hidden_Layer_node_count[NUMHIDDENLAYERS-1], outputLayerBias);

        int guess_label = result_calculator(outputLayer_output, NUMOUTPUTS);
        if(guess_label == training_labels[i])
        {
          correct_answers++;
        }

        // Back Propagation Algorithm /////////////////////////////////////////
        // Calculates Deltas at Output Layer using one-hot representation
        // Treat the output like a Softmax
        // So calculate the sum, which should the output of label node
        if(store_max)
        {
          float sum = 0;
          for(j = 0; j < NUMOUTPUTS; j++)
          {
            sum = sum + outputLayer_output[j];
          }
          what_is_max = sum / NUMOUTPUTS;
          store_max = false;
        }
        // Calculate output gradient
        for(j = 0; j < NUMOUTPUTS; j++)
        {
          if((j + 1) == training_labels[i])
          {
            outputLayer_gradient[j] += (what_is_max - outputLayer_output[j]) * drelu(outputLayer_output[j]);
          }
          else
          {
            outputLayer_gradient[j] += (0 - outputLayer_output[j]) * drelu(outputLayer_output[j]);
          }
        }
        if(((i+1) % BATCH_SIZE) == 0)
        {
          // Normalize Gradients
          for(j = 0; j < NUMOUTPUTS; j++)
          {
            outputLayer_gradient[j] = outputLayer_gradient[j] / BATCH_SIZE;
          }
          // Calculate the gradient for next layer with current weights
          mvmt(outputWeights, outputWeights_gradient, hiddenLayer_gradient[NUMHIDDENLAYERS-1], NUMOUTPUTS, hidden_Layer_node_count[NUMHIDDENLAYERS-1]);

          // Update Output Bias
          for(j = 0; j < NUMOUTPUTS; j++)
          {
            outputLayerBias[j] += LEARN_RATE * outputLayer_gradient[j] / 1000;
          }

          // Update Output Weight
          outer_p(outputLayer_gradient,hiddenLayer_output[NUMHIDDENLAYERS-1],outputWeights_gradient,NUMOUTPUTS,hidden_Layer_node_count[NUMHIDDENLAYERS-1]);
          add_weights_l(outputWeights,outputWeights_gradient,NUMOUTPUTS * hidden_Layer_node_count[NUMHIDDENLAYERS-1]);

          // Calculates Deltas at Hidden Layer
          for(j = (NUMHIDDENLAYERS-1); j >= 0; j--)
          {
            for(k = 0 ; k < hidden_Layer_node_count[NUMHIDDENLAYERS-1]; k++)
            {
              hiddenLayer_gradient[j][k] = hiddenLayer_gradient[j][k] * drelu(hiddenLayer_output[j][k]);
            }

            if(j != 0)
            {
              mvmt(hiddenWeights[j], hiddenLayer_gradient[j], hiddenLayer_gradient[j-1],hidden_Layer_node_count[j],hidden_Layer_node_count[j-1]);
              // Update Hidden Layer Weights
              outer_p(hiddenLayer_gradient[j],hiddenLayer_output[j-1],hiddenWeights_gradient[j],hidden_Layer_node_count[j],hidden_Layer_node_count[j-1]);
              add_weights_l(hiddenWeights[j],hiddenWeights_gradient[j],hidden_Layer_node_count[j] * hidden_Layer_node_count[j-1]);
            }
            else
            {
              outer_p(hiddenLayer_gradient[j],training_inputs[i],hiddenWeights_gradient[j],hidden_Layer_node_count[j],INPUTSIZE);
              add_weights_l(hiddenWeights[j],hiddenWeights_gradient[j],hidden_Layer_node_count[j] * INPUTSIZE);
            }

            // Update Hidden Layer Bias
            for(k = 0; k < hidden_Layer_node_count[j]; k++)
            {
              hiddenLayerBiases[j][k] += LEARN_RATE * hiddenLayer_gradient[j][k] / 1000;
            }



          }
          //RE-Initialize to Zero
          for(j = 0; j < NUMOUTPUTS; j++)
          {
            outputLayer_gradient[j] = 0;
          }
        }
      }
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
      time_stamp = diff(time1,time2);
      printf("Feed Forwarded for Epoch %d\n",iteration);
      printf("Time Taken: %0.2f seconds\nAccuracy: %0.3f\n",(((float)(GIG * time_stamp.tv_sec + time_stamp.tv_nsec)) / GIG), (((float)correct_answers) * 100 / NUMINPUTSTRAIN));
      /*printf("Bias: ");
      for(i = 0; i < NUMOUTPUTS; i++)
      {
        printf("%0.1f,",outputLayerBias[i]);
      }
      printf("\n\nOutput: ");
      for(i = 0; i < NUMOUTPUTS; i++)
      {
        printf("%0.1f,",outputLayer_output[i]);
      }
      printf("\n\n");*/
      //for(i = 0; i < (NUMOUTPUTS * hidden_Layer_node_count[NUMHIDDENLAYERS-1]); i++)
      //{
      //  printf("%0.1f,",outputWeights[i]);
      //}
      //printf("\n\n");
    }


    // Free Initialized Weights/////////////////////////////////////////////////
    for(i = 0; i < NUMHIDDENLAYERS; i++)
    {
      free(hiddenWeights[i]);
      free(hiddenLayer_gradient[i]);
      free(hiddenWeights_gradient[i]);
    }
    free(outputWeights);
    free(outputWeights_gradient);

    printf("Freed Weights\n");
    printf("Freed Gradients\n");
    // Free Neural Network Nodes //////////////////////////////////////////////
    for(i = 0; i < NUMHIDDENLAYERS; i++)
    {
      free(hiddenLayer_output[i]);
    }
    printf("Freed Nodes\n");
  }
  printf("Hello World\n");

  return 0;
}

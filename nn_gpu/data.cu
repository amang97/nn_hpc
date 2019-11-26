/* Copyright 2019, Aman Gupta, Boston University */
#include "data.cuh"
#include <stdio.h>
#include <stdlib.h>
#define BUFFER_SIZE 5120

/* Reading mnist csv batch by batch */
dataset * read_batch(int start, int batch_size, int w, int h, char *file) {
    int i, j, num_features = w*h;
    // create a data set
    dataset * mnist = (dataset *)malloc(sizeof(dataset));
    mnist->n = batch_size;
    mnist->w = w;
    mnist->h = h;

    // create an image
    Pixel *images = (Pixel *)calloc(batch_size*num_features, sizeof(Pixel));
    if (!images) return NULL;
    mnist->images = images;

    // create label array
    int *labels = (int *)calloc(batch_size, sizeof(int));
    if(!labels) return NULL;
    mnist->labels = labels;

    // open image file and start reading data
    FILE * img; char buf[BUFFER_SIZE];
    img = fopen(file, "r");
    if(!img) {printf("Couldn't open image file"); return NULL;}
    fgets(buf, sizeof(buf), img);   // remove first line with headers

    // Read batch_size of data points into data matrix X;
    for(j = start ; j < start+batch_size; j++) {
        fgets(buf, sizeof(buf), img);
        char * tok = strtok(buf, ",");  // split csv by commas
        mnist->labels[j] = atoi(tok);
        for(i = 0; i < num_features; i++)
        {
            tok = strtok(NULL,",");
            if(!tok) {printf("No Input at %d row, %d column\n",j,(i+1)); break;}
            
            data_t pixel = (data_t)atof(tok);
            if(pixel) { pixel = pixel / 255;}   // normalization
            mnist->images[j*num_features+i] = pixel;
        }
    }
    int closed = fclose(img);
    return mnist;
}

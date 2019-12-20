#include "../nn_param.cuh"
#include "../nn/matrix.cuh"
#include "../nn/cuda_utils.cuh"
#include "data.cuh"

data_tr *read_train_img_batch_from(char * filename) {
    data_tr *mnist_tr = (data_tr *)malloc(sizeof(data_tr));
    if (!mnist_tr) { printf("Unable to allocate data_tr\n"); return NULL; }

    // open the image data file
    FILE *images; char buf[BUFFER_SIZE];
    images = fopen(filename, "r");
    if(!images) { printf("Couldn't open image file\n"); return NULL; }
    fgets(buf, sizeof(buf), images);   // remove first line with headers

    // Read Data batch by batch
    int batch_id;
    for (batch_id = 0; batch_id < NUM_BATCHES_TR; batch_id++) {
        mnist_tr->batch[batch_id] = matrix_init(BATCH_SIZE, NUM_FEATURES);
        mnist_tr->label[batch_id] = matrix_init(BATCH_SIZE, NUM_OUTPUTS);
        matrix_allocate(mnist_tr->batch[batch_id]);
        matrix_allocate(mnist_tr->label[batch_id]);

        // fill batches with image data
        int i, j, label, idx_img;
        for(j = 0 ; j < BATCH_SIZE; j++) {
            fgets(buf, sizeof(buf), images);
            char * tok = (char *)strtok(buf, ",");  // split csv by commas
            idx_img = j*NUM_FEATURES;
            label = atoi(tok);
            mnist_tr->label[batch_id]->data_h[j*NUM_OUTPUTS+label] = (data_t)1;
            for(i = 0; i < NUM_FEATURES; i++) {
                tok = strtok(NULL,",");
                if(!tok) { printf("No Input at %d, %d\n",j,(i+1)); break; }
                data_t pixel = (data_t)atof(tok);
                if(pixel) { pixel = pixel / 255;}   // normalization
                mnist_tr->batch[batch_id]->data_h[idx_img+i] = pixel;
            }
        }

        // copy data from host to device
        copy_matrix_H2D(mnist_tr->batch[batch_id]);
        copy_matrix_H2D(mnist_tr->label[batch_id]);
    }
    
    // close the data file
    int closed = fclose(images);
    return mnist_tr;
}

data_tt *read_test_img_batch_from(char * filename) {
    data_tt *mnist_tt = (data_tt *)malloc(sizeof(data_tt));
    if (!mnist_tt) {printf("Unable to allocate data_tt\n"); return NULL; }

    // open the image data file
    FILE *images; char buf[BUFFER_SIZE];
    images = fopen(filename, "r");
    if(!images) {printf("Couldn't open image file\n"); return NULL;}
    fgets(buf, sizeof(buf), images);   // remove first line with headers

    // Read Data batch by batch
    int batch_id;
    for (batch_id = 0; batch_id < NUM_BATCHES_TT; batch_id++) {
        mnist_tt->batch[batch_id] = matrix_init(BATCH_SIZE, NUM_FEATURES);
        mnist_tt->label[batch_id] = matrix_init(BATCH_SIZE, NUM_OUTPUTS);
        matrix_allocate(mnist_tt->batch[batch_id]);
        matrix_allocate(mnist_tt->label[batch_id]);
        
        // fill batches with image data
        int i, j, label, idx_img;
        for(j = 0 ; j < BATCH_SIZE; j++) {
            fgets(buf, sizeof(buf), images);
            char * tok = (char *)strtok(buf, ",");  // split csv by commas
            idx_img = j*NUM_FEATURES;
            label = atoi(tok);
            mnist_tt->label[batch_id]->data_h[j*NUM_OUTPUTS+label] = (data_t)1;
            for(i = 0; i < NUM_FEATURES; i++) {
                tok = strtok(NULL,",");
                if(!tok) { printf("No Input at %d, %d\n",j,(i+1)); break; }
                data_t pixel = (data_t)atof(tok);
                if(pixel) { pixel = pixel / 255;}   // normalization
                mnist_tt->batch[batch_id]->data_h[idx_img+i] = pixel;
            }
        }

        // copy data from host to device
        copy_matrix_H2D(mnist_tt->batch[batch_id]);
        copy_matrix_H2D(mnist_tt->label[batch_id]);
    }
    
    // close the data file
    int closed = fclose(images);
    return mnist_tt;
}

data_tr *load_mnist_train(char * file_with_training_data_in_csv) {
    return read_train_img_batch_from(file_with_training_data_in_csv);
}

data_tt *load_mnist_test(char *file_with_test_data_in_csv) {
    return read_test_img_batch_from(file_with_test_data_in_csv);
}

Matrix *get_batch_data_tr(data_tr *mnist_tr, int batch_id) {
    return mnist_tr->batch[batch_id];
}

Matrix *get_batch_data_tt(data_tt *mnist_tt, int batch_id) {
    return mnist_tt->batch[batch_id];
}

Matrix *get_batch_label_tr(data_tr *mnist_tr, int batch_id) {
    return mnist_tr->label[batch_id];
}

Matrix *get_batch_label_tt(data_tt *mnist_tt, int batch_id) {
    return mnist_tt->label[batch_id];
}

int free_data(data_tr *tr, data_tt *tt) {
    if ((!tr) || (!tt)) return -1;
    int batch_id;
    for (batch_id = 0; batch_id < NUM_BATCHES_TR; batch_id++) {
        free(tr->batch[batch_id]);
        free(tr->label[batch_id]);
    }
    free(tr);
    for (batch_id = 0; batch_id < NUM_BATCHES_TT; batch_id++) {
        free(tt->batch[batch_id]);
        free(tt->label[batch_id]);
    }
    free(tt);
    return 0;
}
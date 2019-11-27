#include "nn_param.cuh"
#include "matrix.cuh"
#include "cuda_utils.cuh"
#include "load_data.cuh"

data_tr *read_train_img_batch_from(char * filename) {
    data_tr *mnist_tr = (data_tr *)malloc(sizeof(data_tr));
    int batch_id;
    for (batch_id = 0; batch_id < NUM_BATCHES_TR; batch_id++) {
        mnist_tr->batch[batch_id] = matrix_init(BATCH_SIZE, NUM_FEATURES);
        mnist_tr->label[batch_id] = matrix_init(BATCH_SIZE, 1);
        matrix_allocate_host(mnist_tr->batch[batch_id]);
        matrix_allocate_cuda(mnist_tr->batch[batch_id]);
        matrix_allocate_host(mnist_tr->label[batch_id]);
        matrix_allocate_cuda(mnist_tr->label[batch_id]);

        // fill batches with image data
        FILE *images; char buf[BUFFER_SIZE];
        images = fopen(file, "r");
        if(!images) {printf("Couldn't open image file"); return NULL;}
        fgets(buf, sizeof(buf), images);   // remove first line with headers
        int i, j, start = batch_id*BATCH_SIZE;
        for(j = start ; j < start+BATCH_SIZE; j++) {
            fgets(buf, sizeof(buf), images);
            char * tok = (char *)strtok(buf, ",");  // split csv by commas
            mnist_tr->label[batch_id]->data_h[j] = atoi(tok);
            for(i = 0; i < NUM_FEATURES; i++) {
                tok = strtok(NULL,",");
                if(!tok) { printf("No Input at %d, %d\n",j,(i+1)); break; }
                data_t pixel = (data_t)atof(tok);
                if(pixel) { pixel = pixel / 255;}   // normalization
                mnist_tr->batch[batch_id]->data_h[j*NUM_FEATURES+i] = pixel;
            }
        }
        int closed = fclose(images);

        // copy data from host to device
        copy_matrix_H2D(mnist_tr->batch[batch_id]);
        copy_matrix_H2D(mnist_tr->label[batch_id]);
    }
    return mnist_tr;
}

data_tt *read_test_img_batch_from(char * filename) {
    data_tt *mnist_tt = (data_tt *)malloc(sizeof(data_tt));
    int batch_id;
    for (batch_id = 0; batch_id < NUM_BATCHES_TR; batch_id++) {
        mnist_tt->batch[batch_id] = matrix_init(BATCH_SIZE, NUM_FEATURES);
        mnist_tt->label[batch_id] = matrix_init(BATCH_SIZE, 1);
        matrix_allocate_host(mnist_tt->batch[batch_id]);
        matrix_allocate_cuda(mnist_tt->batch[batch_id]);
        matrix_allocate_host(mnist_tt->label[batch_id]);
        matrix_allocate_cuda(mnist_tt->label[batch_id]);

        // fill batches with image data
        FILE *images; char buf[BUFFER_SIZE];
        images = fopen(file, "r");
        if(!images) {printf("Couldn't open image file"); return NULL;}
        fgets(buf, sizeof(buf), images);   // remove first line with headers
        int i, j, start = batch_id*BATCH_SIZE;
        for(j = start ; j < start+BATCH_SIZE; j++) {
            fgets(buf, sizeof(buf), images);
            char * tok = (char *)strtok(buf, ",");  // split csv by commas
            mnist_tt->label[batch_id]->data_h[j] = atoi(tok);
            for(i = 0; i < NUM_FEATURES; i++) {
                tok = strtok(NULL,",");
                if(!tok) { printf("No Input at %d, %d\n",j,(i+1)); break; }
                data_t pixel = (data_t)atof(tok);
                if(pixel) { pixel = pixel / 255;}   // normalization
                mnist_tt->batch[batch_id]->data_h[j*NUM_FEATURES+i] = pixel;
            }
        }
        int closed = fclose(images);

        // copy data from host to device
        copy_matrix_H2D(mnist_tt->batch[batch_id]);
        copy_matrix_H2D(mnist_tt->label[batch_id]);
    }
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

Matrix *gat_batch_label_tr(data_tr *mnist_tr, int batch_id) {
    return mnist_tr->label[batch_id];
}

Matrix *gat_batch_label_tt(data_tt *mnist_tt, int batch_id) {
    return mnist_tt->label[batch_id];
}

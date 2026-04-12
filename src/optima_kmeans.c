// Wrapper function to run k-means algorithm on a given dataset
#include "OptimaKmeans/optima_kmeans.h"
#include "dataloader.h"
#include "kmeans.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


int optima_load_data_bin(const char* filename, double** data, int* n, int* d) {
    return load_data_bin(filename, data, n, d);
}

int optima_load_data_csv(const char* filename, double** data, int* n, int* d) {
    return load_data_csv(filename, data, n, d);
}

double* optima_kmeans(double *points, int num_points, int dim, int k, int max_iter, int *clusters) {
    return kmeans(points, num_points, dim, k, max_iter, clusters);
}

void optima_malloc_clusters(int** clusters, int n) {
    *clusters = malloc(n * sizeof(int));
}

void optima_free_data(double* data, double* centroids, int* clusters) {
    free_data(data);
    free(centroids);
    free(clusters);
}

void optima_kmeans_gpu(double *h_data, int n, int d, int k, int max_iter, int *h_clusters, double *h_initial_centroids) {
    double *d_data, *d_centroids;
    int *d_clusters;
    cudaMalloc(&d_data, n * d * sizeof(double));
    cudaMalloc(&d_clusters, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * d * sizeof(double), cudaMemcpyHostToDevice);
    double* d_final_centroids = kmeans_gpu(d_data, n, d, k, max_iter, d_clusters, h_initial_centroids);
    cudaMemcpy(h_clusters, d_clusters, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_clusters);
    cudaFree(d_final_centroids);
}
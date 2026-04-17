// Wrapper function to run k-means algorithm on a given dataset
#include "OptimaKmeans/optima_kmeans.h"
#include "dataloader.h"
#include "kmeans.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//extern
double* kmeans_gpu(double *h_data, int num_points, int dim, int k, int max_iteration,
                   int *h_clusters, int *finished_iterations, int threads_per_block);
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

KMeansResult optima_kmeans_gpu(double *data, int num_points, int dim, int k, int max_iteration, int *clusters) {
    KMeansResult result;
    result.centroids = kmeans_gpu(data, num_points, dim, k, max_iteration, clusters, &result.iterations, 256);
    return result;
}

KMeansResult optima_kmeans_gpu_threads(double *data, int num_points, int dim, int k, int max_iteration,
                                       int *clusters, int threads_per_block) {
    KMeansResult result;
    result.centroids = kmeans_gpu(data, num_points, dim, k, max_iteration, clusters, &result.iterations, threads_per_block);
    return result;
}
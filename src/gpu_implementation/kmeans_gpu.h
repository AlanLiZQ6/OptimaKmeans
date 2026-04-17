#ifndef KMEANS_GPU_H
#define KMEANS_GPU_H

#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

#ifdef __cplusplus
extern "C" {
#endif

// Finding the nearest centroid for every point
__global__ void find_centroid(double *d_data, double *d_centroids, int *d_clusters, 
                                  int N, int D, int K);

// Calculating the centroids
__global__ void centroid_sum(double *d_data, int *d_clusters, double *d_new_centroids, int *d_counts, int N, int D, int K);
__global__ void calculate_centroid(double *d_new_centroids, int *d_counts, int D, int K);

// Host-side GPU entry point. Returns centroid array with size K * D.
double* kmeans_gpu(double *h_data, int num_points, int dim, int k, int max_iteration,
                   int *h_clusters, int *finished_iterations, int threads_per_block);

#ifdef __cplusplus
}
#endif

#endif
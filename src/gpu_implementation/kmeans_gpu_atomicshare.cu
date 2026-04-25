#include "kmeans_gpu.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Assign points to nearest centroid and track if any assignments changed
 *
 * @param device_data                 Stores the data by column major order on the GPU
 * @param device_centroids            Stores the centroids by row major order on the GPU
 * @param device_clusters             Stores the cluster assignments of points on the GPU
 * @param device_cluster_changed      The signal that indicates if any point has new cluster in the current iteration
 * @param N                           The number of points
 * @param dimensions                  The number of dimensions
 * @param K                           The number of clusters
 */
__global__ void find_centroid(double *device_data, double *device_centroids, int *device_clusters, int *device_cluster_changed, int N, int dimensions, int K)
{
    extern __shared__ double shared_centroids[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    for (int i = tid; i < K * dimensions; i += blockDim.x) {
        shared_centroids[i] = device_centroids[i];
    }
    __syncthreads();

    if (idx < N) {
        double min_distance = 1e18;
        int closest_centroid = 0;

        for (int k = 0; k < K; k++) {
            double current_distance = 0.0;
            for (int d = 0; d < dimensions; ++d) {
                double diff = device_data[idx + (d * N)] - shared_centroids[k * dimensions + d];
                current_distance += diff * diff;
            }
            if (current_distance < min_distance) {
                min_distance = current_distance;
                closest_centroid = k;
            }
        }

        if (device_clusters[idx] != closest_centroid) {
            device_clusters[idx] = closest_centroid;
            *device_cluster_changed = 1;
        }
    }
}

/**
 * Calculates the sum of points in each cluster
 * Each thread atomically accumulates its point into shared memory,
 * then flushes shared memory to global memory.
 *
 * @param device_data The input data by column major order
 * @param device_clusters The cluster assignments of points
 * @param device_new_sums The space to store the new sums of points in each cluster
 * @param device_num_point_each_cluster The array to store the count of points in each cluster
 * @param N The number of points
 * @param dimensions The dimensions of data
 * @param K The number of clusters
 */
__global__ void centroid_sum(double *device_data, int *device_clusters, double *device_new_sums, int *device_num_point_each_cluster, int N, int dimensions, int K)
{
    extern __shared__ unsigned char shared_buffer[];

    int *shared_counts = (int *)shared_buffer;
    size_t counts_bytes = (size_t)K * sizeof(int);
    size_t sums_offset = (counts_bytes + sizeof(double) - 1) & ~(sizeof(double) - 1);
    double *shared_sums = (double *)(shared_buffer + sums_offset);

    int tid  = threadIdx.x;
    int idx  = blockIdx.x * blockDim.x + tid;

    for (int i = tid; i < K; i += blockDim.x) {
        shared_counts[i] = 0;
    }

    for (int i = tid; i < K * dimensions; i += blockDim.x) {
        shared_sums[i] = 0.0;
    }

    __syncthreads();

    if (idx < N) {
        int cluster_id = device_clusters[idx];

        if (cluster_id >= 0 && cluster_id < K) {
            for (int d = 0; d < dimensions; d++) {
                double v = device_data[idx + (d * N)];
                atomicAdd(&shared_sums[cluster_id * dimensions + d], v);
            }
            atomicAdd(&shared_counts[cluster_id], 1);
        }
    }

    __syncthreads();

    for (int i = tid; i < K; i += blockDim.x) {
        int count = shared_counts[i];
        if (count > 0) {
            atomicAdd(&device_num_point_each_cluster[i], count);
        }
    }

    for (int i = tid; i < K * dimensions; i += blockDim.x) {
        double val = shared_sums[i];
        if (val != 0.0) {
            atomicAdd(&device_new_sums[i], val);
        }
    }
}

/**
 * Calculates the new centroids for each cluster
 * @param device_new_sums The space to store the new sums of points in each cluster
 * @param device_num_point_each_cluster The array to store the count of points in each cluster
 * @param dimensions The dimensions of the data
 * @param K The number of clusters
 */
__global__ void calculate_centroid(double *device_new_sums, int *device_num_point_each_cluster, int dimensions, int K)
{
    int cid = blockIdx.x;
    int d = threadIdx.x;

    if (cid >= K || d >= dimensions) {
        return;
    }

    int count = device_num_point_each_cluster[cid];
    if (count > 0) {
        device_new_sums[cid * dimensions + d] /= count;
    }
}

/**
 * Performs K-Means clustering on the GPU
 * @param h_data The input data array
 * @param num_points The number of data points
 * @param dimension The dimension of the data
 * @param k The number of clusters
 * @param max_iteration The maximum number of iterations
 * @param host_clusters The output cluster assignments
 * @param finished_iterations The number of iterations completed before convergence
 * @return A pointer to the final centroids
 */
double* kmeans_gpu(double *h_data, int num_points, int dimension, int k, int max_iteration,
                   int *host_clusters, int *finished_iterations, int threads_per_block)
{
    double *host_initial_centroids = (double *)malloc((size_t)k * dimension * sizeof(double));
    double *host_data_column_major_order = (double *)malloc((size_t)num_points * dimension * sizeof(double));

    for (int i = 0; i < num_points; i++) {
        for (int d = 0; d < dimension; d++) {
            if (i < k) {
                host_initial_centroids[i * dimension + d] = h_data[i * dimension + d];
            }
            host_data_column_major_order[d * num_points + i] = h_data[i * dimension + d];
        }
    }

    memset(host_clusters, -1, num_points * sizeof(int));

    double *device_data;
    double *device_centroids;
    double *device_new_sums;
    int *device_clusters;
    int *device_num_point_each_cluster;
    int *device_cluster_changed;

    cudaMalloc(&device_data, num_points * dimension * sizeof(double));
    cudaMalloc(&device_centroids, k * dimension * sizeof(double));
    cudaMalloc(&device_new_sums, k * dimension * sizeof(double));
    cudaMalloc(&device_clusters, num_points * sizeof(int));
    cudaMalloc(&device_num_point_each_cluster, k * sizeof(int));
    cudaMalloc(&device_cluster_changed, sizeof(int));

    cudaMemcpy(device_data, host_data_column_major_order, num_points * dimension * sizeof(double), cudaMemcpyHostToDevice);
    free(host_data_column_major_order);

    cudaMemcpy(device_centroids, host_initial_centroids, k * dimension * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_clusters, host_clusters, num_points * sizeof(int), cudaMemcpyHostToDevice);

    int thread_per_block = (threads_per_block > 0) ? threads_per_block : 256;
    int blocksPerGrid_Points = (num_points + thread_per_block - 1) / thread_per_block;
    size_t shared_mem_size = (size_t)k * dimension * sizeof(double);

    int iteration;
    int host_changed;

    for (iteration = 0; iteration < max_iteration; iteration++) {
        host_changed = 0;
        cudaMemcpy(device_cluster_changed, &host_changed, sizeof(int), cudaMemcpyHostToDevice);

        find_centroid<<<blocksPerGrid_Points, thread_per_block, shared_mem_size>>>(
            device_data, device_centroids, device_clusters, device_cluster_changed, num_points, dimension, k);
        cudaDeviceSynchronize();

        cudaMemcpy(&host_changed, device_cluster_changed, sizeof(int), cudaMemcpyDeviceToHost);

        if (host_changed == 0) {
            break;
        }

        cudaMemset(device_num_point_each_cluster, 0, k * sizeof(int));
        cudaMemset(device_new_sums, 0, k * dimension * sizeof(double));

        size_t shared_counts_bytes = (size_t)k * sizeof(int);
        size_t shared_sums_offset = (shared_counts_bytes + sizeof(double) - 1) & ~(sizeof(double) - 1);
        size_t centroid_sum_shared_mem = shared_sums_offset + (size_t)k * dimension * sizeof(double);

        centroid_sum<<<blocksPerGrid_Points, thread_per_block, centroid_sum_shared_mem>>>(
            device_data, device_clusters, device_new_sums, device_num_point_each_cluster, num_points, dimension, k);
        cudaDeviceSynchronize();

        calculate_centroid<<<k, dimension>>>(device_new_sums, device_num_point_each_cluster, dimension, k);
        cudaDeviceSynchronize();

        cudaMemcpy(device_centroids, device_new_sums, k * dimension * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    if (finished_iterations != NULL) {
        *finished_iterations = iteration;
    }

    double *host_final_centroids = (double *)malloc((size_t)k * dimension * sizeof(double));
    cudaMemcpy(host_final_centroids, device_centroids, k * dimension * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_clusters, device_clusters, num_points * sizeof(int), cudaMemcpyDeviceToHost);

    free(host_initial_centroids);
    cudaFree(device_data);
    cudaFree(device_centroids);
    cudaFree(device_new_sums);
    cudaFree(device_clusters);
    cudaFree(device_num_point_each_cluster);
    cudaFree(device_cluster_changed);

    return host_final_centroids;
}
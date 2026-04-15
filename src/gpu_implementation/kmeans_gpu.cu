#include "kmeans_gpu.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Assign points to nearest centroid and track if any assignments changed
 *
 * @param d_data         Stores the data by row-major with [N * D]
 * @param d_centroids    Stores the centroids by row-major with [K * D]
 * @param d_clusters     Stores the cluster ID for each point with [N]
 * @param d_changed      Stores a flag indicating if any assignments changed with [1]
 * @param N              The number of points
 * @param D              The number of dimensions
 * @param K              The number of clusters 
 */
__global__ void find_centroid(double *d_data, double *d_centroids, int *d_clusters, int *d_changed, int N, int D, int K)
{
    
    // The shared memory has the current centroids for this iteration, so all threads in the block can access
    extern __shared__ double shared_centroids[];
    // The thread ID
    int tid = threadIdx.x;
    // The global ID of the thread in the grid
    int idx = blockIdx.x * blockDim.x + tid;

    // Load current centroids into shared memory
    for (int i = tid; i < K * D; i += blockDim.x) {
        shared_centroids[i] = d_centroids[i];
    }
    __syncthreads();

    if (idx < N) {
        double min_distance = 1e18;
        int closest_centroid = 0;

        for (int k = 0; k < K; k++) {
            // Calculate the Euclidean distance between two points
            double current_distance = 0.0;
            for (int d = 0; d < D; d++) {
                // We use row-major indexing
                double diff = d_data[idx * D + d] - shared_centroids[k * D + d];
                current_distance += diff * diff;
            }
            current_distance = sqrt(current_distance);
            if (current_distance < min_distance) {
                min_distance = current_distance;
                closest_centroid = k;
            }
        }

        // If the centroid of a point has changed changed, we mark it and update the cluster in the next step
        if (d_clusters[idx] != closest_centroid) {
            d_clusters[idx] = closest_centroid;
            *d_changed = 1; 
        }
    }
}

/**
 * @brief Sum coordinates and update the number of points in each cluster so that we can compute new centroids in the next step
 *
 * @param d_data         Stores the data by row-major with [N * D]
 * @param d_clusters     Stores the cluster ID for each point with [N]
 * @param d_new_sums     Stores the sum of coordinates for each cluster with [K * D]
 * @param d_counts       Stores the number of points in each cluster with [K]
 * @param N              The number of points
 * @param D              The number of dimensions
 */
__global__ void centroid_sum(double *d_data, int *d_clusters, double *d_new_sums, int *d_counts, int N, int D)
{
    // For each thread, it will process one data point and update the corresponding cluster's sum and count
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // The cluster ID of the current data point
        int cluster_id = d_clusters[idx];
        // If this cluster ID is valid
        if (cluster_id >= 0) {
            // Atomically add the point's coordinates to the cluster's sum and increment the count
            atomicAdd(&d_counts[cluster_id], 1);
            for (int d = 0; d < D; d++) {
                atomicAdd(&d_new_sums[cluster_id * D + d], d_data[idx * D + d]);
            }
        }
    }
}

/**
* @brief Update centroids by dividing the sum of coordinates by the count of points in each cluster
*
* @param d_centroids    Stores the centroids by row-major with [K * D]
* @param d_new_sums     Stores the sum of coordinates for each cluster with [K * D]
* @param d_counts       Stores the number of points in each cluster with [K]
* @param K              The number of clusters 
* @param D              The number of dimensions
*/
__global__ void update_centroids(double *d_centroids, double *d_new_sums, int *d_counts, int K, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread will update one coordinate of one centroid
    if (idx < K * D) {
        int cluster_id = idx / D;
        int count = d_counts[cluster_id];
        if (count > 0) {
            d_centroids[idx] = d_new_sums[idx] / (double)count;
        }
    }
}

double* kmeans_gpu(double *h_data, int num_points, int dim, int k, int max_iteration, int *h_clusters, int *finished_iterations)
{
    double *host_initial_centroids = (double *)malloc((size_t)k * dim * sizeof(double));
    for (int i = 0; i < k; i++) {
        for (int d = 0; d < dim; d++) {
            host_initial_centroids[i * dim + d] = h_data[i * dim + d];
        }
    }
    memset(h_clusters, -1, num_points * sizeof(int));

    double *device_data, *device_centroids, *device_new_sums;
    int *device_clusters, *device_counts, *device_changed;
    
    cudaMalloc(&device_data, num_points * dim * sizeof(double));
    cudaMalloc(&device_centroids, k * dim * sizeof(double));
    cudaMalloc(&device_new_sums, k * dim * sizeof(double));
    cudaMalloc(&device_clusters, num_points * sizeof(int));
    cudaMalloc(&device_counts, k * sizeof(int));
    cudaMalloc(&device_changed, sizeof(int));

    cudaMemcpy(device_data, h_data, num_points * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_centroids, host_initial_centroids, k * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_clusters, h_clusters, num_points * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid_Points = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid_Centroids = (k * dim + threadsPerBlock - 1) / threadsPerBlock;
    size_t shared_mem_size = k * dim * sizeof(double);

    int iter;
    int h_changed;

    for (iter = 0; iter < max_iteration; iter++) {
        h_changed = 0;
        cudaMemcpy(device_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

        find_centroid<<<blocksPerGrid_Points, threadsPerBlock, shared_mem_size>>>(
            device_data, device_centroids, device_clusters, device_changed, num_points, dim, k);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, device_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_changed == 0) {
            break;
        }

        cudaMemset(device_counts, 0, k * sizeof(int));
        cudaMemset(device_new_sums, 0, k * dim * sizeof(double));

        centroid_sum<<<blocksPerGrid_Points, threadsPerBlock>>>(
            device_data, device_clusters, device_new_sums, device_counts, num_points, dim);
        cudaDeviceSynchronize();

        update_centroids<<<blocksPerGrid_Centroids, threadsPerBlock>>>(
            device_centroids, device_new_sums, device_counts, k, dim);
        cudaDeviceSynchronize();
    }

    if (finished_iterations != NULL) {
        *finished_iterations = iter;
    }

    double *h_final_centroids = (double *)malloc(k * dim * sizeof(double));
    cudaMemcpy(h_final_centroids, device_centroids, k * dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clusters, device_clusters, num_points * sizeof(int), cudaMemcpyDeviceToHost);

    free(host_initial_centroids);
    cudaFree(device_data);
    cudaFree(device_centroids);
    cudaFree(device_new_sums);
    cudaFree(device_clusters);
    cudaFree(device_counts);
    cudaFree(device_changed);

    return h_final_centroids;
}
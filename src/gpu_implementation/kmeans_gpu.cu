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
    // The shared memory has the current centroids for this iteration, so all threads in the block can access
    extern __shared__ double shared_centroids[];
    // The thread ID
    int tid = threadIdx.x;
    // The global ID of the thread in the grid
    int idx = blockIdx.x * blockDim.x + tid;

    // Load current centroids into shared memory
    // It is row major order, example: centroid 1 dim 1, centroid 1 dim 2, ..., centroid 1 dim dimensions, centroid 2 dim 1, ...
    for (int i = tid; i < K * dimensions; i += blockDim.x) {
        shared_centroids[i] = device_centroids[i];
    }
    __syncthreads();

    // A thread will find the closest centroid for one data point
    if (idx < N) {
        double min_distance = 1e18;
        int closest_centroid = 0;

        for (int k = 0; k < K; k++) {
            // Calculate the squared Euclidean distance of each dimension between the point and the centroid and calculate the total distance
            // The euclidean distance is sqrt(sum((point dimension - centroid dimension)^2)), since we only care about the relative distance, 
            // we do not need the sqrt and just calculate sum((point dimension - centroid dimension)^2)
            double current_distance = 0.0;
            for (int d = 0; d < dimensions; ++d) {
                // device is column major order and shared memory is row major order
                double diff = device_data[idx + (d * N)] - shared_centroids[k * dimensions + d];
                current_distance += diff * diff;
            }
            // Update the closest centroid if total distance is smaller
            if (current_distance < min_distance) {
                min_distance = current_distance;
                closest_centroid = k;
            }
        }

        // If the centroid of a point has changed, we mark it and update the cluster in the next step
        if (device_clusters[idx] != closest_centroid) {
            device_clusters[idx] = closest_centroid;
            *device_cluster_changed = 1; 
        }
    }
}

/**
 * Calculates the sum of points in each cluster
 * @param device_data The input data data by column major order
 * @param device_clusters The cluster assignments of points
 * @param device_new_sums The space to store the new sums of points in each cluster
 * @param device_num_point_each_cluster The array to store the count of points in each cluster
 * @param N The number of points
 * @param dimensions The dimensions of data
 * @param K The number of clusters
 */
__global__ void centroid_sum(double *device_data, int *device_clusters, double *device_new_sums, int *device_num_point_each_cluster, int N, int dimensions, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    int cluster_id = device_clusters[idx];

    if (cluster_id < 0 || cluster_id >= K) {
        return;
    }

    unsigned int active = __activemask();
    unsigned int group = __match_any_sync(active, cluster_id);
    int lane = threadIdx.x & 31;
    int leader = __ffs(group) - 1;

    // TODO: write comment
    for (int d = 0; d < dimensions; d++) {
        double v = device_data[idx + (d * N)]; 
        for (int offset = 16; offset > 0; offset >>= 1) {
            double other = __shfl_down_sync(active, v, offset);
            int partner = lane + offset;
            if (partner < 32 && ((group >> partner) & 1)) {
                v += other;
            }
        }
        if (lane == leader) {
            atomicAdd(&device_new_sums[cluster_id * dimensions + d], v);
        }
    }

    if (lane == leader) {
        atomicAdd(&device_num_point_each_cluster[cluster_id], __popc(group));
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
    // Get the count of points in this cluster
    int count = device_num_point_each_cluster[cid];
    // We can calculate the new centroid by dividing the sum of points in this cluster by the count
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
    // The space to store the initial centroids
    double *host_initial_centroids = (double *)malloc((size_t)k * dimension * sizeof(double));
    // The space to store the data in column major order for better memory access on the GPU
    // example: point 1 dim 1, point 2 dim 1, ..., point N dim 1, point 1 dim 2, ...
    // When we load the data from csv, the data is in row major order
    // example: point 1 dim 1, point 1 dim 2, ..., point 1 dim dimensions, point 2 dim 1, ...
    double *host_data_column_major_order = (double *)malloc((size_t)num_points * dimension * sizeof(double));

    for (int i = 0; i < num_points; i++) {
        for (int d = 0; d < dimension; d++) {
            if (i < k) {
                // Takes the first k point in data as the initial centroids
                host_initial_centroids[i * dimension + d] = h_data[i * dimension + d];
            }
            // Transpose the data from row major to column major
            host_data_column_major_order[d * num_points + i] = h_data[i * dimension + d];
        }
    }
    
    // Initialize the cluster assignments to -1 which means unassigned
    memset(host_clusters, -1, num_points * sizeof(int));

    // The space to store data on the GPU
    double *device_data;
    // The space to store centroids on the GPU
    double *device_centroids;
    // The space to store the new sums of points in each cluster on the GPU
    double *device_new_sums;
    // The space to store cluster assignments of points on the GPU
    int *device_clusters;
    // The space to store the count of points in each cluster on the GPU
    int *device_num_point_each_cluster;
    // The signal that indicates if any point has new cluster in the current iteration on the GPU
    int *device_cluster_changed;
    
    cudaMalloc(&device_data, num_points * dimension * sizeof(double));
    cudaMalloc(&device_centroids, k * dimension * sizeof(double));
    cudaMalloc(&device_new_sums, k * dimension * sizeof(double));
    cudaMalloc(&device_clusters, num_points * sizeof(int));
    cudaMalloc(&device_num_point_each_cluster, k * sizeof(int));
    cudaMalloc(&device_cluster_changed, sizeof(int));

    // Copy the column-major data to the GPU and free it
    cudaMemcpy(device_data, host_data_column_major_order, num_points * dimension * sizeof(double), cudaMemcpyHostToDevice);
    free(host_data_column_major_order);

    // Copy centroids and initial clusters
    cudaMemcpy(device_centroids, host_initial_centroids, k * dimension * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_clusters, host_clusters, num_points * sizeof(int), cudaMemcpyHostToDevice);

    int thread_per_block = (threads_per_block > 0) ? threads_per_block : 256;
    // Number of blocks GPU needed to lanuch. We would like all points are covered by threads 
    int blocksPerGrid_Points = (num_points + thread_per_block - 1) / thread_per_block;
    size_t shared_mem_size = k * dimension * sizeof(double);

    int iteration;
    // The flag to indicate if any point has new cluster in the current iteration on CPU
    int host_changed;

    // Main Kmeans Loop
    for (iteration = 0; iteration < max_iteration; iteration++) {
        // Set the flag to 0 that means no point has new cluster in the current iteration
        host_changed = 0;
        // Copy the flag to GPU
        cudaMemcpy(device_cluster_changed, &host_changed, sizeof(int), cudaMemcpyHostToDevice);

        // Find the closest centroid for each point and update the cluster assignment, if any point has new cluster, set the flag to 1
        find_centroid<<<blocksPerGrid_Points, thread_per_block, shared_mem_size>>>(
            device_data, device_centroids, device_clusters, device_cluster_changed, num_points, dimension, k);
        cudaDeviceSynchronize();

        // Copy the flag back to CPU
        cudaMemcpy(&host_changed, device_cluster_changed, sizeof(int), cudaMemcpyDeviceToHost);

        // If no point has new cluster, we can stop the iteration
        if (host_changed == 0) {
            break;
        }

        // Initialize the space to store the sums of points in each cluster and the count of points in each cluster to 0 before calculating the new centroids
        cudaMemset(device_num_point_each_cluster, 0, k * sizeof(int));
        cudaMemset(device_new_sums, 0, k * dimension * sizeof(double));


        // Calculate the sum of points in each cluster and the count of points in each cluster
        centroid_sum<<<blocksPerGrid_Points, thread_per_block>>>(
            device_data, device_clusters, device_new_sums, device_num_point_each_cluster, num_points, dimension, k);
        cudaDeviceSynchronize();

        // Calculate the new centroids by dividing the sums by the count of points in each cluster.
        // We edit device_new_sums in place so it is the same as the new centroids
        calculate_centroid<<<k, dimension>>>(device_new_sums, device_num_point_each_cluster, dimension, k);
        cudaDeviceSynchronize();

        // Copy the updated values back to device_centroids for the next iteration
        cudaMemcpy(device_centroids, device_new_sums, k * dimension * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    if (finished_iterations != NULL) {
        *finished_iterations = iteration;
    }

    // Copy results back to CPU
    double *host_final_centroids = (double *)malloc(k * dimension * sizeof(double));
    cudaMemcpy(host_final_centroids, device_centroids, k * dimension * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_clusters, device_clusters, num_points * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    free(host_initial_centroids);
    cudaFree(device_data);
    cudaFree(device_centroids);
    cudaFree(device_new_sums);
    cudaFree(device_clusters);
    cudaFree(device_num_point_each_cluster);
    cudaFree(device_cluster_changed);

    return host_final_centroids;
}
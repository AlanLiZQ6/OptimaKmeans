#include "kmeans_gpu.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// Finding the nearest centroid for every point
__global__ void find_centroid(double *d_data, double *d_centroids, int *d_clusters, int N, int D, int K)
{
    //TODO
    // Load current centroids in shared memory
    extern __shared__ double shared_centroids[];
    // The thread id in the thread block
    int tid = threadIdx.x;
    // The thread id in the grid
    int idx = blockIdx.x * blockDim.x + tid;

    for (int i = tid; i < K *D; i += blockDim.x) {
        shared_centroids[i] = d_centroids[i];
    }

    // Make sure all threads load centroids in shared memory
    __syncthreads();

    if (idx < N) {
        double smallest_distance = DBL_MAX;
        int closest_centroid = -1;

        for (int k = 0; k < K; k++) {
            double current_distance = 0.0;
            for (int d = 0; d < D; d++) {
                // Calculates the Euclidean distance between two points
                double point_val = d_data[idx + (d * N)];
                double centroid_val = shared_centroids[k * D + d];
                current_distance += (point_val - centroid_val) * (point_val - centroid_val);
            }
            if (current_distance < smallest_distance) {
                smallest_distance = current_distance;
                closest_centroid = k;
            }
        }
        d_clusters[idx] = closest_centroid;
    }
}

// Calculating the centroids
__global__ void centroid_sum(double *d_data, int *d_clusters, double *d_new_centroids, int *d_counts, int N, int D, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int cid = d_clusters[idx];
    if (cid < 0 || cid >= K) return;
    unsigned int active = __activemask();
    unsigned group = __match_any_sync(active, cid);
    int lane = threadIdx.x & 31;
    int leader = __ffs(group) - 1;
    for (int d = 0; d < D; ++d) {
        double v = d_data[idx + (d * N)];
        for (int offset = 16; offset > 0; offset >>= 1) {
            double other = __shfl_down_sync(active, v, offset);
            int partner = lane + offset;
            if (partner < 32 && ((group >> partner) & 1))
                v += other;
        }
        if (lane==leader) atomicAdd(&d_new_centroids[cid * D + d], v);
    }
    if (lane==leader) atomicAdd(&d_counts[cid], __popc(group));
}

__global__ void calculate_centroid(double *d_new_centroids, int *d_counts, int D, int K)
{
    int cid = blockIdx.x;
    int d = threadIdx.x;
    if (cid >= K || d >= D) return;
    int count = d_counts[cid];
    if (count > 0) {
        d_new_centroids[cid * D + d] /= count;
    }
    
}

double* kmeans_gpu(double *d_data, int num_points, int dim, int k, int max_iteration, int *d_clusters, double *h_initial_centroids)
{
    double *d_new_centroids;
    double *d_centroids;
    int *d_counts;

    cudaMalloc(&d_new_centroids, k * dim * sizeof(double));
    cudaMalloc(&d_centroids, k * dim * sizeof(double));
    cudaMalloc(&d_counts, k * sizeof(int));

    cudaMemcpy(d_centroids, h_initial_centroids, k * dim * sizeof(double), cudaMemcpyHostToDevice);
    
    size_t shared_mem_size = k * dim * sizeof(double);
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < max_iteration; iter++) {
        find_centroid<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_data, d_centroids, d_clusters, num_points, dim, k);
        cudaDeviceSynchronize();
        cudaMemset(d_new_centroids, 0, k * dim * sizeof(double));
        cudaMemset(d_counts, 0, k * sizeof(int));
        centroid_sum<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_clusters, d_new_centroids, d_counts, num_points, dim, k);
        cudaDeviceSynchronize();
        calculate_centroid<<<k, dim>>>(d_new_centroids, d_counts, dim, k);
        cudaDeviceSynchronize();
        cudaMemcpy(d_centroids, d_new_centroids, k * dim * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_counts);
    cudaFree(d_new_centroids);
    
    return d_centroids;
}
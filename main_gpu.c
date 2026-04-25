// example main.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "OptimaKmeans/optima_kmeans_gpu.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <percentage>\n", argv[0]);
        return -1;
    }

    double percentage = atof(argv[1]);

    double* data;
    int n, d;

    // Load data from a csv file
    if (optima_load_data_csv("/afs/ece.cmu.edu/usr/zhuoqili/Private/OptimaKmeans/dataset/data/f1_data/processed/final_processed.csv", &data, &n, &d) != 0) {
        fprintf(stderr, "Failed to load data from CSV file\n");
        return 1;
    }

    // Use only a percentage of the data
    n = (int)(n * percentage);

    // Run k-means
    int k = 5;
    int max_iter = 500;
    int* clusters;
    optima_malloc_clusters(&clusters, n);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    KMeansResult gpu_result = optima_kmeans_gpu(data, n, d, k, max_iter, clusters);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_sec = (double)(t1.tv_sec - t0.tv_sec) +
                         (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    double time_ms = elapsed_sec * 1e3;
    int iters = gpu_result.iterations;

    double* centroids = gpu_result.centroids;
    double inertia = 0.0;
    for (int i = 0; i < n; i++) {
        int c = clusters[i];
        double sum_sq = 0.0;
        for (int j = 0; j < d; j++) {
            double diff = data[i * d + j] - centroids[c * d + j];
            sum_sq += diff * diff;
        }
        inertia += sum_sq;
    }

    printf("Time: %.2f ms, Iterations: %d, Time per Iteration: %.2f ms, Inertia: %.6f\n",
           time_ms, iters, iters > 0 ? time_ms / iters : 0.0, inertia);

#ifdef DEBUG
    // Print centroids
    for (int i = 0; i < k; i++) {
        printf("Centroid %d: ", i);
        for (int j = 0; j < d; j++) {
            printf("%f ", centroids[i * d + j]);
        }
        printf("\n");
    }
    // Print cluster assignments
    printf("Cluster assignments:\n");
    for (int i = 0; i < n; i++) {
        printf("Point %d assigned to cluster %d\n", i, clusters[i]);
    }
#endif

    // Free allocated memory
    optima_free_data(data, gpu_result.centroids, clusters);

    return 0;
}

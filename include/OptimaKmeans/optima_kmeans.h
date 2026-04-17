// Header file for OptimaKmeans library
#ifndef OPTIMA_KMEANS_H
#define OPTIMA_KMEANS_H

typedef struct {
    double* centroids;
    int iterations;
} KMeansResult;

int optima_load_data_bin(const char* filename, double** data, int* n, int* d);

double* kmeans_gpu(double *h_data, int num_points, int dim, int k, int max_iteration,
                   int *h_clusters, int *finished_iterations, int threads_per_block);

int optima_load_data_csv(const char* filename, double** data, int* n, int* d);

double* optima_kmeans(double *points, int num_points, int dim, int k, int max_iter, int *clusters);

void optima_free_data(double* data, double* centroids, int* clusters);

void optima_malloc_clusters(int** clusters, int n);

KMeansResult optima_kmeans_gpu(double *data, int num_points, int dim, int k, int max_iteration, int *clusters);

KMeansResult optima_kmeans_gpu_threads(double *data, int num_points, int dim, int k, int max_iteration,
                                       int *clusters, int threads_per_block);

#endif // OPTIMA_KMEANS_H
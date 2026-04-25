#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kmeans.h"
#include <string.h>
#include <omp.h>

static float dist_sq(float *p1, float *p2, int dim)
{
    float sum = 0;
    for (int i = 0; i < dim; i++)
    {
        sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return sum;
}

float *kmeans(float *data, int num_points, int dim, int k, int max_iteration, int *clusters, int *iter_converge)
{
    float *centroids = malloc(k * dim * sizeof(float));
    int *counts = malloc(k * sizeof(int));
    float *new_sums = malloc(k * dim * sizeof(float));
    memset(clusters, -1, num_points * sizeof(int));
    int centroid_changed = 0;
    *iter_converge = max_iteration;

    // Per-thread scratch buffers allocated once; zeroed each iteration.
    int nthreads = omp_get_max_threads();
    float *all_local_sums = malloc(nthreads * k * dim * sizeof(float));
    int *all_local_counts = malloc(nthreads * k * sizeof(int));

    // Initial centroids: first k data points (matches kmeans.c).
    for (int i = 0; i < k; i++)
    {
        for (int d = 0; d < dim; d++)
        {
            centroids[i * dim + d] = data[i * dim + d];
        }
    }

    
        for (int iter = 0; iter < max_iteration; iter++)
        {
            
            centroid_changed = 0;
            memset(counts, 0, k * sizeof(int));
            memset(new_sums, 0, k * dim * sizeof(float));
            
            #pragma omp parallel
            {
                // --- Assignment step ---
                #pragma omp for reduction(| : centroid_changed)
                for (int i = 0; i < num_points; i++)
                {
                    float min_d = 1e18f;
                    int closest_centroid = 0;
                    for (int centroid = 0; centroid < k; centroid++)
                    {
                        float distance = dist_sq(&data[i * dim], &centroids[centroid * dim], dim);
                        if (distance < min_d)
                        {
                            min_d = distance;
                            closest_centroid = centroid;
                        }
                    }
                    if (clusters[i] != closest_centroid)
                    {
                        clusters[i] = closest_centroid;
                        centroid_changed = 1;
                    }
                }
            }

            if (!centroid_changed)
            {
                *iter_converge = iter;
                printf("Converged at iteration %d\n", iter);
                break;
            }

            // --- Update step (centroid sums & counts) ---
            // First Version : each thread accumulates into its own slice,
            // then one thread serially merges all slices.
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                float *local_sums = &all_local_sums[tid * k * dim];
                int *local_counts = &all_local_counts[tid * k];
                memset(local_sums, 0, k * dim * sizeof(float));
                memset(local_counts, 0, k * sizeof(int));

                #pragma omp for schedule(static)
                for (int i = 0; i < num_points; i++)
                {
                    int cid = clusters[i];
                    for (int d = 0; d < dim; d++)
                        local_sums[cid * dim + d] += data[i * dim + d];
                    local_counts[cid]++;
                }

                #pragma omp single
                {
                    for (int t = 0; t < nthreads; t++)
                    {
                        for (int c = 0; c < k; c++)
                        {
                            counts[c] += all_local_counts[t * k + c];
                            for (int d = 0; d < dim; d++)
                                new_sums[c * dim + d] += all_local_sums[t * k * dim + c * dim + d];
                        }
                    }
                }
            
                // --- Recompute centroids from sums/counts ---
                #pragma omp for schedule(static)
                for (int centroid = 0; centroid < k; centroid++)
                {
                    if (counts[centroid] > 0)
                    {
                        for (int d = 0; d < dim; d++)
                            centroids[centroid * dim + d] = new_sums[centroid * dim + d] / counts[centroid];
                    }
                }
            }
        }

    free(new_sums);
    free(counts);
    free(all_local_sums);
    free(all_local_counts);
    return centroids;
}

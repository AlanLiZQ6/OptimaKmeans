#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "kmeans.h"
#include <string.h>

/**
 * @brief Calculates the Euclidean distance between two points
 *
 * @param p1 Point one's feature vector
 * @param p2 Point two's feature vector
 * @param dim Number of dimensions (features) per point
 * @return double The Euclidean distance between the two points
 */
static double dist(double *p1, double *p2, int dim)
{
    double sum = 0;
    // Euclidean distance: sqrt((x1-x2)^2 + (y1-y2)^2 + ...)
    for (int i = 0; i < dim; i++)
    {
        sum += (p1[i] - p2[i])*(p1[i] - p2[i]);
    }
    return sqrt(sum);
}

double *kmeans(double *data, int num_points, int dim, int k, int max_iteration, int *clusters)
{
    // Allocate memory for centroids
    // centroids is a 1D array [k * dim]
    double *centroids = malloc(k * dim * sizeof(double));

    memset(clusters, -1, num_points * sizeof(int));

    // Pick random points as starting centroids
    // srand(time(NULL));
    for (int i = 0; i < k; i++)
    {
        int r = rand() % num_points;
        for (int d = 0; d < dim; d++)
        {
            centroids[i * dim + d] = data[r * dim + d];
        }
    }

    for (int iter = 0; iter < max_iteration; iter++)
    {
        int centroid_changed = 0;

        // Find closest centroid for each point
        for (int i = 0; i < num_points; i++)
        {
            double min_d = 1e18;
            // The index of the closest centroid for each point
            int closest_centroid = 0;
            // Calculate distance of this point to each centroid and find the closest one
            for (int centroid = 0; centroid < k; centroid++)
            {
                double distance = dist(&data[i * dim], &centroids[centroid * dim], dim);
                if (distance < min_d)
                {
                    min_d = distance;
                    closest_centroid = centroid;
                }
            }
            // If the closest centroid is different from the current cluster assignment, update it
            if (clusters[i] != closest_centroid)
            {
                clusters[i] = closest_centroid;
                centroid_changed = 1;
            }
        }

        // If no points changed clusters, we have converged
        if (!centroid_changed)
        {
            break;
        }

        // Calculate new centroids
        int *counts = calloc(k, sizeof(int));
        double *new_sums = calloc(k * dim, sizeof(double));

        // Sum up the coordinates of points in each cluster to calculate the new centroids
        for (int i = 0; i < num_points; i++)
        {
            for (int d = 0; d < dim; d++) {
                new_sums[clusters[i] * dim + d] += data[i * dim + d];
            }
            counts[clusters[i]]++;
        }

        // Update centroids by calculating the mean of the points assigned to each cluster
        for (int centroid = 0; centroid < k; centroid++)
        {
            if (counts[centroid] > 0)
            {
                for (int d = 0; d < dim; d++)
                    centroids[centroid * dim + d] = new_sums[centroid * dim + d] / counts[centroid];
            }
        }
        free(new_sums);
        free(counts);
    }
    return centroids;
}
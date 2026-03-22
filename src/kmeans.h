#ifndef KMEANS_H
#define KMEANS_H

/**
 * @param points: Array of data points, where each point is represented as a flat array of features (size: num_points * dim)
 * @param num_points: Total points in the dataset
 * @param dim: Number of dimensions (features) per point
 * @param k: Number of clusters to find
 * @param max_iteration: Maximum iterations to run the algorithm
 * @param clusters: Array of the cluster assignment for each point
 * @return: A 2D array of the final centroids [k][dim]
 */
double** kmeans(double *points, int num_points, int dim, int k, int max_iter, int *clusters);

#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "OptimaKmeans/optima_kmeans.h"

typedef struct {
    const char *data_path;
    const char *variant;
    int k;
    int max_iter;
    int threads;
} Options;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [--data PATH] [--k INT] [--max-iter INT] "
            "[--variant NAME] [--threads INT]\n"
            "\n"
            "Examples:\n"
            "  %s --data ../data/final_processed.csv --k 8 --variant baseline --threads 256\n",
            prog,
            prog);
}

static int parse_int_arg(const char *value, int *out) {
    char *end = NULL;
    long v = strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return -1;
    }
    if (v < -2147483647L || v > 2147483647L) {
        return -1;
    }
    *out = (int)v;
    return 0;
}

static int parse_args(int argc, char **argv, Options *opt) {
    int i;
    opt->data_path = "../data/final_processed.csv";
    opt->variant = "baseline";
    opt->k = 5;
    opt->max_iter = 10000;
    opt->threads = 256;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            opt->data_path = argv[++i];
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &opt->k) != 0 || opt->k <= 0) {
                fprintf(stderr, "Invalid --k value\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--max-iter") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &opt->max_iter) != 0 || opt->max_iter <= 0) {
                fprintf(stderr, "Invalid --max-iter value\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            opt->variant = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &opt->threads) != 0 || opt->threads <= 0) {
                fprintf(stderr, "Invalid --threads value\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 1;
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return -1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    Options opt;
    double *data = NULL;
    int *clusters = NULL;
    int n = 0;
    int d = 0;
    int parse_status;

    parse_status = parse_args(argc, argv, &opt);
    if (parse_status != 0) {
        return (parse_status > 0) ? 0 : 2;
    }

    if (optima_load_data_csv(opt.data_path, &data, &n, &d) != 0) {
        fprintf(stderr, "Failed to load data from CSV file: %s\n", opt.data_path);
        return 1;
    }

    if (strcmp(opt.variant, "baseline") != 0) {
        fprintf(stderr,
                "Note: --variant=%s is accepted for tuning metadata, but current binary exposes one GPU implementation.\n",
                opt.variant);
    }

    optima_malloc_clusters(&clusters, n);

    {
        struct timespec t0;
        struct timespec t1;
        double elapsed_sec;
        double inertia;
        KMeansResult gpu_result;
        int i;
        int j;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        gpu_result = optima_kmeans_gpu_threads(data, n, d, opt.k, opt.max_iter, clusters, opt.threads);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        elapsed_sec = (double)(t1.tv_sec - t0.tv_sec) +
                      (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

        printf("[cfg] variant=%s D=%d K=%d threads=%d N=%d\n",
               opt.variant,
               d,
               opt.k,
               opt.threads,
               n);
        printf("elapsed: %.6f s (%.3f ms), iterations: %d\n",
               elapsed_sec,
               elapsed_sec * 1e3,
               gpu_result.iterations);

        inertia = 0.0;
        for (i = 0; i < n; i++) {
            int c = clusters[i];
            if (c < 0) {
                continue;
            }
            for (j = 0; j < d; j++) {
                /* N-major indexing: point i, dimension j */
                double diff = data[i * d + j] - gpu_result.centroids[c * d + j];
                inertia += diff * diff;
            }
        }
        printf("inertia: %.10f\n", inertia);

        optima_free_data(NULL, gpu_result.centroids, NULL);
    }

    optima_free_data(data, NULL, clusters);
    return 0;
}

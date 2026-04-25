# OptimaKmeans
### To reproduce the dataset, please run the command line by line

## Dataset Download

### Download F1 telemetry data
### We only use the data from 2018 to 2024 from the race Monza
### Make sure you are in the directory OptimaKmeans/dataset
python data_download.py --year_range 2018-2024 --race Monza --session R

### Feature Extraction
python feature_extract.py

### Data Preprocess
python data_preprocess.py


### In this way, the processed data will be in the directory ../OptimaKmeans/dataset/data/f1_data/processed/final_processed.csv, which will be used in the project.


## Baseline 

### For Baseline, we can directly to run and record all the result by using the .sh file
./baseline_sklearn/run_py.sh

### Also, you can run the specific percentage of data
### python baseline_sklearn/kmeans_base.py percentage (0.1, 0.2, ..., 1)
python baseline_sklearn/kmeans_base.py 1

### After you run the run_py.sh, we can run this command to get the plot for baseline
python baseline_sklearn/time_plot.py







**How To Run Example**
mkdir build
cd build
cmake ..
cmake --build . --config Release
./example

## How to run benchmark
Benchmarks use the `autotune` target (GPU timing on a CSV dataset) and an optional Python driver that sweeps kernel variants and thread-block sizes.
mkdir build
cd build
cmake ..
cmake --build . --config Release
benchmark run:
python3 scripts/benchmark.py with options
- `--csvpath`  pass dataset file path
- `--variants` limit the experiment approach such as 'cpu, gpu`
- `--repeats`  runs per configuration (default `3`)

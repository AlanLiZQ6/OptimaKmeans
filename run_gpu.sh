set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# GPU version: base | warp | warpmem | warpmemfound | atomicshare  (default: warp)
GPU_VERSION="${1:-warp}"

cd "$SCRIPT_DIR"
rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TARGET=gpu -DGPU_VERSION="${GPU_VERSION}" ..
make

cd "$SCRIPT_DIR"
OUTPUT="results_gpu_${GPU_VERSION}.csv"
echo "percentage,time,iters,time per iterations,inertia" > "$OUTPUT"

for percentage in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "=== [${GPU_VERSION}] Running with ${percentage} of data ==="
    output=$(./build/example "$percentage")
    line=$(echo "$output" | grep "Time:")
    time_ms=$(echo "$line" | sed -n 's/.*Time: \([0-9.]*\) ms.*/\1/p')
    iters=$(echo "$line" | sed -n 's/.*Iterations: \([0-9]*\).*/\1/p')
    tpi=$(echo "$line" | sed -n 's/.*Time per Iteration: \([0-9.]*\) ms.*/\1/p')
    inertia=$(echo "$line" | sed -n 's/.*Inertia: \([0-9.]*\).*/\1/p')
    echo "${percentage},${time_ms},${iters},${tpi},${inertia}" >> "$OUTPUT"
done

echo "Results saved to $OUTPUT"

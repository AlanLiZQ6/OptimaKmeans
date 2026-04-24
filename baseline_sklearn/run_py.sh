set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"
OUTPUT="external_baseline/results_py.csv"
echo "percentage,time,iters,time_per_iter,inertia" > "$OUTPUT"

for percentage in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "=== Running with ${percentage} of data ==="
    out=$(python external_baseline/kmeans_base.py "$percentage")
    echo "$out"
    time_ms=$(echo "$out"      | sed -n 's/^total time: \([0-9.]*\) ms/\1/p')
    iters=$(echo "$out"        | sed -n 's/^iterations: \([0-9]*\)/\1/p')
    tpi=$(echo "$out"          | sed -n 's/^time per iteration: \([0-9.]*\) ms/\1/p')
    inertia=$(echo "$out"      | sed -n 's/^inertia: \([0-9.eE+-]*\)/\1/p')
    echo "${percentage},${time_ms},${iters},${tpi},${inertia}" >> "$OUTPUT"
done

echo "Results saved to $OUTPUT"

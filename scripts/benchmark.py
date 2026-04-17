#!/usr/bin/env python3
"""Auto-enumeration benchmark driver for GPU K-means tuning.

This script enumerates (variant, threadsPerBlock), runs a command template,
parses execution time, and prints per-configuration timing.
"""

import argparse
import csv
import itertools
import os
import re
import statistics
import subprocess
import sys
import time


def parse_int_list(raw):
	vals = []
	for x in raw.split(','):
		x = x.strip()
		if x:
			vals.append(int(x))
	return vals


def parse_str_list(raw):
	vals = []
	for x in raw.split(','):
		x = x.strip()
		if x:
			vals.append(x)
	return vals


def parse_time_ms(output_text, regex):
	m = re.search(regex, output_text, flags=re.MULTILINE)
	if not m:
		return None
	try:
		return float(m.group(1))
	except Exception:
		return None


def parse_inertia(output_text):
	m = re.search(r"^\s*inertia:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*$", output_text, flags=re.MULTILINE)
	if not m:
		return None
	try:
		return float(m.group(1))
	except Exception:
		return None


def parse_cfg_meta(output_text):
	# Expected line from autotune binary:
	# [cfg] variant=baseline D=7 K=5 threads=256 N=3800047
	m = re.search(r"\[cfg\].*D=(\d+).*K=(\d+).*threads=(\d+).*N=(\d+)", output_text)
	if not m:
		return None
	return {
		"d": int(m.group(1)),
		"k": int(m.group(2)),
		"threads": int(m.group(3)),
		"n": int(m.group(4)),
	}


def probe_hardware(repo_root):
	script = os.path.join(repo_root, "scripts", "autotune_probe.py")
	cmd = [sys.executable, script]
	proc = subprocess.run(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		universal_newlines=True,
		check=False,
	)
	if proc.returncode != 0:
		return None, "probe-failed", proc.stderr.strip()
	try:
		import json
		obj = json.loads(proc.stdout)
		return obj, "probe", ""
	except Exception:
		return None, "probe-invalid-json", proc.stdout[-300:]


def infer_thread_limits(hw):
	default_max = 1024
	default_warp = 32
	if not isinstance(hw, dict):
		return default_max, default_warp, "default"
	gpus = hw.get("gpu", {}).get("gpus", [])
	if not gpus:
		return default_max, default_warp, "default"
	first = gpus[0]
	max_val = first.get("max_threads_per_block", 0)
	warp_val = first.get("warp_size", 0)
	try:
		max_val = int(max_val)
	except Exception:
		max_val = 0
	try:
		warp_val = int(warp_val)
	except Exception:
		warp_val = 0
	if max_val <= 0:
		max_val = default_max
	if warp_val <= 0:
		warp_val = default_warp
	return max_val, warp_val, "autotune-probe"


def generate_thread_candidates(max_threads, warp_size):
	"""Generate thread candidates from hardware limits without a fixed list."""
	if max_threads <= 0:
		return [128]

	min_threads = warp_size if warp_size > 0 else 32
	top_pow2 = 1 << (int(max_threads).bit_length() - 1)
	vals = []
	cur = top_pow2
	while cur >= min_threads:
		vals.append(cur)
		cur //= 2

	if not vals:
		return [max_threads]

	vals.reverse()
	return vals


def auto_threads_from_hardware(hw):
	"""Return thread candidates constrained by inferred hardware limits."""
	max_threads, warp_size, source = infer_thread_limits(hw)
	choices = generate_thread_candidates(max_threads, warp_size)
	return choices, max_threads, warp_size, source


def run_once(cmd, timeout_sec):
	start = time.time()
	proc = subprocess.run(
		cmd,
		shell=True,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		universal_newlines=True,
		timeout=timeout_sec,
		check=False,
	)
	elapsed = (time.time() - start) * 1000.0
	return proc.returncode, proc.stdout, elapsed


def build_runner_command(template, variant, threads, csvpath):
	"""Build runner command and auto-inject data/variant/threads when omitted."""
	cmd = template.format(
		variant=variant,
		threads=threads,
		csvpath=csvpath,
	)

	if "--data" not in cmd:
		cmd = "{} --data {}".format(cmd, csvpath)
	if "--variant" not in cmd:
		cmd = "{} --variant {}".format(cmd, variant)
	if "--threads" not in cmd:
		cmd = "{} --threads {}".format(cmd, threads)
	return cmd


def ensure_parent(path):
	parent = os.path.dirname(path)
	if parent and not os.path.exists(parent):
		os.makedirs(parent)


def main():
	parser = argparse.ArgumentParser(description="Auto-enumeration benchmark for K-means tuning")
	parser.add_argument(
		"--runner-cmd-template",
		default="./build/autotune",
		help=(
			"Command template to execute one trial. Supports placeholders: "
			"{variant} {threads} {csvpath}. If omitted, benchmark auto-appends "
			"'--data <csvpath> --variant <v> --threads <t>'. Example: "
			"'./build/autotune --k 5'"
		),
	)
	parser.add_argument(
		"--csvpath",
		default="./data/final_processed.csv",
		help="Input dataset CSV path for autotune (default: ./data/final_processed.csv)",
	)
	parser.add_argument("--variants", default="baseline,find_warp", help="Comma-separated variants")
	parser.add_argument(
		"--threads",
		default="auto",
		help="Comma-separated threadsPerBlock values, or 'auto' to infer from GPU hardware",
	)
	parser.add_argument("--repeats", type=int, default=3, help="Repeats per configuration")
	parser.add_argument("--timeout-sec", type=int, default=1800, help="Timeout per run")
	parser.add_argument(
		"--time-regex",
		default=r"\(([0-9]+(?:\.[0-9]+)?)\s*ms\)",
		help="Regex with one capture group for time in ms",
	)
	parser.add_argument(
		"--csv-out",
		default="scripts/autotune_trials.csv",
		help="CSV output path for all trial timings",
	)
	parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

	args = parser.parse_args()

	variants = parse_str_list(args.variants)
	repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	hw, probe_source, probe_err = probe_hardware(repo_root)
	if hw is None:
		print("[warn] hardware probe unavailable ({}): {}".format(probe_source, probe_err))
		print("[warn] fallback limits: maxThreadsPerBlock=1024 warpSize=32")

	if args.threads.strip().lower() == "auto":
		tbs, max_threads, warp_size, max_source = auto_threads_from_hardware(hw)
		threads_source = "hardware-auto"
	else:
		tbs = parse_int_list(args.threads)
		max_threads, warp_size, max_source = infer_thread_limits(hw)
		threads_source = "user-specified"
		tbs = [x for x in tbs if x <= max_threads]
		if not tbs:
			print("[warn] no user-specified threads <= maxThreadsPerBlock, fallback to hardware-auto")
			tbs = generate_thread_candidates(max_threads, warp_size)

	csv_out = os.path.join(repo_root, args.csv_out)
	ensure_parent(csv_out)

	total_cfg = len(variants) * len(tbs)
	print("[info] total configurations: {}".format(total_cfg))
	print("[info] repeats per configuration: {}".format(args.repeats))
	print("[info] inferred maxThreadsPerBlock ({}): {}".format(max_source, max_threads))
	print("[info] inferred warpSize ({}): {}".format(max_source, warp_size))
	print("[info] csvpath: {}".format(args.csvpath))
	print("[info] threads ({}) : {}".format(threads_source, ",".join(str(x) for x in tbs)))
	print("[info] csv_out: {}".format(csv_out))

	csv_rows = []
	cfg_index = 0

	for variant, threads in itertools.product(variants, tbs):
		cfg_index += 1
		try:
			cmd = build_runner_command(args.runner_cmd_template, variant, threads, args.csvpath)
		except KeyError as exc:
			print("[error] runner-cmd-template contains unsupported placeholder: {}".format(exc))
			print("[error] supported placeholders are: {variant}, {threads}, {csvpath}")
			return 2

		print("[cfg {}/{}] variant={} threads={}".format(
			cfg_index, total_cfg, variant, threads
		))
		print("[cmd] {}".format(cmd))

		if args.dry_run:
			continue

		times_ms = []
		detected_n = None
		detected_d = None
		detected_k = None
		detected_threads = None
		for rep in range(args.repeats):
			rc, out, wall_ms = run_once(cmd, args.timeout_sec)
			parsed_ms = parse_time_ms(out, args.time_regex)
			parsed_inertia = parse_inertia(out)
			meta = parse_cfg_meta(out)

			if rc != 0:
				print("[trial] rep={} rc={} (failed)".format(rep + 1, rc))
				print(out[-1200:])
				continue

			if parsed_ms is None:
				print("[trial] rep={} rc=0 but time parse failed".format(rep + 1))
				print(out[-1200:])
				continue

			times_ms.append(parsed_ms)
			if parsed_inertia is None:
				print("[trial] rep={} time_ms={:.6f} wall_ms={:.2f} inertia=NA".format(rep + 1, parsed_ms, wall_ms))
			else:
				print("[trial] rep={} time_ms={:.6f} wall_ms={:.2f} inertia={:.10f}".format(rep + 1, parsed_ms, wall_ms, parsed_inertia))
			if meta is not None:
				detected_n = meta["n"]
				detected_d = meta["d"]
				detected_k = meta["k"]
				detected_threads = meta["threads"]

			csv_rows.append(
				{
					"variant": variant,
					"d": (detected_d if detected_d is not None else ""),
					"k": (detected_k if detected_k is not None else ""),
					"threads": (detected_threads if detected_threads is not None else threads),
					"rep": rep + 1,
					"time_ms": "{:.6f}".format(parsed_ms),
					"wall_ms": "{:.2f}".format(wall_ms),
					"inertia": ("{:.10f}".format(parsed_inertia) if parsed_inertia is not None else ""),
				}
			)

		if not times_ms:
			print("[cfg] no successful timing samples, skipped cache record")
			continue

		best_ms = min(times_ms)
		med_ms = statistics.median(times_ms)
		print("[cfg] done best_ms={:.6f} median_ms={:.6f}".format(best_ms, med_ms))
		if detected_n is None or detected_d is None or detected_k is None:
			print("[warn] could not parse D/K/N from runner output")

	if not args.dry_run:
		with open(csv_out, "w", newline="") as f:
			writer = csv.DictWriter(
				f,
				fieldnames=["variant", "d", "k", "threads", "rep", "time_ms", "wall_ms", "inertia"],
			)
			writer.writeheader()
			for row in csv_rows:
				writer.writerow(row)
		print("[done] wrote {} trial rows to {}".format(len(csv_rows), csv_out))


if __name__ == "__main__":
	main()
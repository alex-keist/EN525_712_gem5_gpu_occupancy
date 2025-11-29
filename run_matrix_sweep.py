#!/usr/bin/env python3
import subprocess
from pathlib import Path

def main():
    # Path to repo root (assume script exists in repo root)
    root = Path(__file__).resolve().parent

    # Docker image to use
    img = "ghcr.io/gem5/gcn-gpu:v24-0"

    # List of (width, block_x, block_y) configs to sweep
    configs = [
        (1024, 16, 16),
        (1024, 8, 8),
        (1024, 16, 8),
        (1024, 4, 4),
        (1024, 32, 32),
        (1024, 32, 16),
        (1024, 32, 8),
        (1024, 32, 4),
        #(2048, 16, 16),
    ]

    for width, block_x, block_y in configs:
        run_name = f"W{width}_BX{block_x}_BY{block_y}"
        outdir = root / f"m5out_{run_name}_v9analyzer"

        print("=" * 60)
        print(f"Running config: width={width}, block_x={block_x}, block_y={block_y}")
        print(f"Output dir:     {outdir}")
        print("=" * 60)

        outdir.mkdir(parents=True, exist_ok=True)

        # run gem5 inside docker with a unique outdir
        gem5_cmd = [
            "docker", "run", "--rm",
            "-v", f"{root}:{root}",
            "-w", str(root),
            img,
            "gem5/build/VEGA_X86/gem5.opt",
            "-re", f"--outdir={outdir}",
            "gem5/configs/example/apu_se.py",
            "-n", "3",
            "--gfx-version=gfx902",
            "-c", "gem5-resources/src/gpu/hip-samples/bin/MatrixTranspose_configurable",
            "-o", f"{width} {block_x} {block_y}",
        ]

        print("[*] Running gem5...")
        subprocess.run(gem5_cmd, check=True)

        stats_path = outdir / "stats.txt"
        if not stats_path.is_file():
            print(f"[!] WARNING: {stats_path} not found, skipping analysis for {run_name}")
            continue

        # run the occupancy analyzer for this run
        print("[*] Analyzing stats...")
        analyzer_script = root / "analyzer_dir/gpu_occupancy_analyzer_v9.py"
        analyzer_cmd = [
            "python3",
            str(analyzer_script),
            str(stats_path)
        ]

        # run analyzer with cwd=outdir so the generated image lands in that directory
        subprocess.run(analyzer_cmd, check=True, cwd=outdir)

        print(f"[✓] Done with {run_name}\n")

    print("All configurations completed.")

if __name__ == "__main__":
    main()

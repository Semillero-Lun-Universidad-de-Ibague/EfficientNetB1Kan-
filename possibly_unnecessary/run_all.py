import os, sys, subprocess, concurrent.futures


def run_script(script, directory, gpu_id):
    print(f"\nRunning {script} on GPU {gpu_id}:")
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    try:
        subprocess.run([sys.executable, os.path.join(directory, script)],
                       check=True, env=env)
        print(f"Finished running {script}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")


def run_python_scripts(directory='.'):
    python_files = [f for f in os.listdir(directory) if f.endswith('.py') and f.startswith("test_")]

    if not python_files:
        print(f"No Python scripts found in {directory}")
        return

    print(f"Found {len(python_files)} Python scripts. Running them in pairs...")


    workers=1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for i in range(0, len(python_files), workers):
            futures = []
            for j in range(workers):
                if i + j < len(python_files):
                    futures.append(executor.submit(run_script, python_files[i + j], directory, j))

            concurrent.futures.wait(futures)


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    run_python_scripts(directory)
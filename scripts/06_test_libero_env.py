"""
06_test_libero_env.py
Verify LIBERO simulation environment can render headlessly on cloud GPU.
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

print("Step 1: Import LIBERO...")
from libero.libero import benchmark
print("  ✅ LIBERO imported")

print("\nStep 2: Load LIBERO-Spatial benchmark...")
benchmark_dict = benchmark.get_benchmark_dict()
print(f"  Available benchmarks: {list(benchmark_dict.keys())}")

task_suite = benchmark_dict["libero_spatial"]()
n_tasks = task_suite.n_tasks
print(f"  ✅ LIBERO-Spatial loaded: {n_tasks} tasks")

print("\nStep 3: Inspect first 3 tasks...")
for i in range(min(3, n_tasks)):
    task = task_suite.get_task(i)
    print(f"  Task {i}: {task.name}")
    print(f"    Language: {task.language}")

print("\nStep 4: Create env + render one frame (the critical headless test)...")
from libero.libero.envs import OffScreenRenderEnv

task = task_suite.get_task(0)
task_bddl_file = os.path.join(
    benchmark.get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
)

env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
}
env = OffScreenRenderEnv(**env_args)
print("  ✅ Environment created")

obs = env.reset()
print(f"  ✅ env.reset() succeeded")
print(f"  Observation keys: {list(obs.keys())}")

agentview = obs.get("agentview_image")
if agentview is not None:
    print(f"  ✅ Render works! agentview shape: {agentview.shape}, dtype: {agentview.dtype}")
else:
    print(f"  ⚠️  No agentview_image in obs")

# Save the rendered frame so we can actually see what LIBERO looks like
import numpy as np
from PIL import Image
if agentview is not None:
    os.makedirs("/root/autodl-tmp/vla/results", exist_ok=True)
    Image.fromarray(agentview).save("/root/autodl-tmp/vla/results/06_libero_first_frame.png")
    print(f"  ✅ Saved first rendered frame to results/06_libero_first_frame.png")

env.close()
print("\n�� LIBERO environment fully functional on this server!")

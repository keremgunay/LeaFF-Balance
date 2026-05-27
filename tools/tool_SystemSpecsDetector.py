"""
check_env.py
------------
Quick system check for the lab PC.
Tells us what we are working with before running any GAN/CNN training.

Just run:
    python check_env.py

No installation needed. Uses only the standard library + whatever is
already installed (torch, tensorflow are checked but optional).
"""

import os
import sys
import platform
import shutil
import subprocess


def header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def run(cmd):
    """Run a shell command and return its stdout, or None if it fails."""
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


# ---------- 1. OS & CPU ----------
header("1. OPERATING SYSTEM")
print(f"System   : {platform.system()}")
print(f"Release  : {platform.release()}")
print(f"Version  : {platform.version()}")
print(f"Machine  : {platform.machine()}")
print(f"Processor: {platform.processor() or 'n/a'}")

# More detail on Linux
if platform.system() == "Linux":
    distro = run("lsb_release -ds") or run("cat /etc/os-release | head -2")
    if distro:
        print(f"Distro   : {distro}")


# ---------- 2. PYTHON ----------
header("2. PYTHON")
print(f"Python version : {sys.version.split()[0]}")
print(f"Python path    : {sys.executable}")
print(f"Implementation : {platform.python_implementation()}")

# Detect virtual env
in_venv = (
    hasattr(sys, "real_prefix")
    or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    or os.environ.get("VIRTUAL_ENV") is not None
    or os.environ.get("CONDA_DEFAULT_ENV") is not None
)
print(f"In virtualenv  : {in_venv}")
if os.environ.get("VIRTUAL_ENV"):
    print(f"  VIRTUAL_ENV    = {os.environ['VIRTUAL_ENV']}")
if os.environ.get("CONDA_DEFAULT_ENV"):
    print(f"  CONDA env name = {os.environ['CONDA_DEFAULT_ENV']}")
if os.environ.get("CONDA_PREFIX"):
    print(f"  CONDA_PREFIX   = {os.environ['CONDA_PREFIX']}")


# ---------- 3. PACKAGE MANAGERS ----------
header("3. PACKAGE MANAGERS AVAILABLE")
for tool in ["pip", "pip3", "conda", "mamba", "uv", "poetry"]:
    path = shutil.which(tool)
    print(f"{tool:8s}: {path if path else 'not found'}")


# ---------- 4. GPU (NVIDIA / AMD) ----------
header("4. GPU HARDWARE")

nvidia_smi = shutil.which("nvidia-smi")
rocm_smi = shutil.which("rocm-smi")

if nvidia_smi:
    print(">>> NVIDIA GPU detected (nvidia-smi found)")
    info = run("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv")
    if info:
        print(info)
    cuda_ver = run("nvidia-smi | grep -oP 'CUDA Version: \\K[0-9.]+'")
    if cuda_ver:
        print(f"CUDA driver version: {cuda_ver}")
else:
    print("nvidia-smi : not found (no NVIDIA GPU, or driver not installed)")

if rocm_smi:
    print("\n>>> AMD GPU detected (rocm-smi found)")
    info = run("rocm-smi --showproductname --showmeminfo vram")
    if info:
        print(info)
else:
    print("rocm-smi   : not found (no AMD ROCm install)")

# Fallback: list all GPUs from lspci on Linux
if platform.system() == "Linux":
    lspci = run("lspci | grep -Ei 'vga|3d|display'")
    if lspci:
        print("\nlspci VGA/3D devices:")
        print(lspci)


# ---------- 5. CUDA / ROCm TOOLKITS ----------
header("5. CUDA / ROCm TOOLKITS (compiler-level)")

nvcc = shutil.which("nvcc")
if nvcc:
    ver = run("nvcc --version")
    print(f"nvcc found at: {nvcc}")
    if ver:
        print(ver)
else:
    print("nvcc       : not found (CUDA toolkit not installed system-wide; PyTorch may still work via its own bundled CUDA)")

cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
print(f"CUDA_HOME  : {cuda_home or 'not set'}")

rocm_path = os.environ.get("ROCM_PATH")
print(f"ROCM_PATH  : {rocm_path or 'not set'}")
if os.path.isdir("/opt/rocm"):
    print("/opt/rocm  : exists")


# ---------- 6. DEEP LEARNING FRAMEWORKS ----------
header("6. DEEP LEARNING FRAMEWORKS")

# PyTorch
try:
    import torch
    print(f"PyTorch version    : {torch.__version__}")
    print(f"  built with CUDA  : {torch.version.cuda}")
    if hasattr(torch.version, "hip"):
        print(f"  built with HIP   : {torch.version.hip}")
    print(f"  cuda.is_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device count     : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} | "
                  f"{props.total_memory / 1024**3:.1f} GB | "
                  f"compute {props.major}.{props.minor}")
    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps"):
        print(f"  mps.is_available : {torch.backends.mps.is_available()}")
except ImportError:
    print("PyTorch    : NOT installed")

# TensorFlow
try:
    import tensorflow as tf
    print(f"\nTensorFlow version : {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPUs visible to TF: {len(gpus)}")
    for g in gpus:
        print(f"  {g}")
except ImportError:
    print("\nTensorFlow : NOT installed")

# Common helper libs (just version, no fuss)
print("\nCommon libraries:")
for pkg in ["numpy", "pandas", "matplotlib", "torchvision",
            "PIL", "cv2", "sklearn", "tqdm"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"  {pkg:12s}: {ver}")
    except ImportError:
        print(f"  {pkg:12s}: not installed")


# ---------- 7. DISK / RAM ----------
header("7. RESOURCES")
try:
    import shutil as _sh
    total, used, free = _sh.disk_usage(os.path.expanduser("~"))
    print(f"Home disk total : {total / 1024**3:.1f} GB")
    print(f"Home disk free  : {free  / 1024**3:.1f} GB")
except Exception as e:
    print(f"Disk check failed: {e}")

# RAM (Linux/Mac via /proc or sysctl, fallback to psutil if installed)
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"RAM total       : {vm.total / 1024**3:.1f} GB")
    print(f"RAM available   : {vm.available / 1024**3:.1f} GB")
except ImportError:
    if platform.system() == "Linux":
        meminfo = run("grep -E 'MemTotal|MemAvailable' /proc/meminfo")
        if meminfo:
            print(meminfo)
    else:
        print("(install psutil for RAM info)")

print("\nDone.\n")
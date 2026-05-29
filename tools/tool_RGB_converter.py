from PIL import Image
import os

src = r"C:\Users\cagla\Desktop\Kerem Günay\hey"
dst = r"C:\Users\cagla\Desktop\Kerem Günay\StyleGAN2_project\datasets_raw\severe_nutdef_1000"
os.makedirs(dst, exist_ok=True)

# Pre-count how many image files we'll process so we can show "X / total"
all_files = [f for f in os.listdir(src)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))]
total = len(all_files)
print(f"Found {total} images to convert.")

count = 0
skipped = 0
for fname in all_files:
    try:
        img = Image.open(os.path.join(src, fname)).convert("RGB")
        out_name = os.path.splitext(fname)[0] + ".png"
        img.save(os.path.join(dst, out_name))
        count += 1

        # Notify every 50 successfully converted images
        if count % 50 == 0:
            print(f"  ...converted {count} / {total} images", flush=True)

    except Exception as e:
        skipped += 1
        print(f"Skipped {fname}: {e}")

print(f"Done. Converted {count} images, skipped {skipped}.")

r'''
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
INITIALIZATION:


Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64

& "C:\Users\cagla\miniconda3\shell\condabin\conda-hook.ps1"
conda activate stylegan2ada

cd "C:\Users\cagla\Desktop\Kerem Günay\StyleGAN2_project\stylegan2-ada-pytorch"

$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;$env:PATH"
$env:TORCH_CUDA_ARCH_LIST = "8.9"

cl
nvcc --version
where.exe nvcc

python -c "import torch, numpy as np; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('numpy:', np.__version__); print('available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
SMOKE TEST

python train.py --outdir="C:\Users\cagla\Desktop\Kerem Günay\StyleGAN2_project\runs" --data="C:\Users\cagla\Desktop\Kerem Günay\StyleGAN2_project\datasets_zip\healthy_low_leaf_4000_256.zip" --gpus=1 --cfg=paper256 --mirror=1 --aug=ada --target=0.6 --snap=1 --metrics=none --kimg=1 --workers=4



REAL RUN

python train.py `
  --outdir="C:\Users\cagla\Desktop\Kerem Günay\results_school" `
  --data="C:\Users\cagla\Desktop\Kerem Günay\StyleGAN2_project\datasets_zip\flies_up_back_leaf_2000_256.zip" `
  --gpus=1 `
  --cfg=paper256 `
  --batch=16 `
  --mirror=1 `
  --aug=ada `
  --target=0.6 `
  --kimg=1000 `
  --snap=10 `
  --metrics=fid50k_full
  
'''
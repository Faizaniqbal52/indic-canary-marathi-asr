import builtins
import os
import sys

orig_open = builtins.open

def utf8_open(file, mode='r', *args, **kwargs):
    if 'w' in mode and 'b' not in mode and 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return orig_open(file, mode, *args, **kwargs)

builtins.open = utf8_open

from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()
    kernel = sys.argv[1] if len(sys.argv) > 1 else 'ac2sny/indic-canary-marathi-lora-fine-tuning'
    target_dir = sys.argv[2] if len(sys.argv) > 2 else 'kaggle/output_finetune'
    os.makedirs(target_dir, exist_ok=True)
    print(f"Fetching outputs for {kernel} into {target_dir}...")
    outfiles, token = api.kernels_output(kernel, path=target_dir, force=True, quiet=False)
    print(f"Downloaded {len(outfiles)} files:")
    for f in outfiles:
        print("  -", f)

if __name__ == '__main__':
    main()

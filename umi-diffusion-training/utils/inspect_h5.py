#!/usr/bin/env python3
import h5py
import sys
import numpy as np

h5_path = sys.argv[1] if len(sys.argv) > 1 else 'umi-diffusion-training/data/umi-pick-cube/DAS-Gripper_20260320190303_none_none_00000000_vio.h5'

with h5py.File(h5_path, 'r') as f:
    def show(name, obj):
        if isinstance(obj, h5py.Group):
            print(f'[Group] {name}')
            if len(obj.attrs) > 0:
                print(f'  attrs: {dict(obj.attrs)}')

        elif isinstance(obj, h5py.Dataset):
            print(f'[Dataset] {name}')
            print(f'  shape: {obj.shape}')
            print(f'  dtype: {obj.dtype}')

            if len(obj.attrs) > 0:
                print(f'  attrs: {dict(obj.attrs)}')

            # 小数据直接显示内容
            try:
                if obj.size <= 10:
                    data = obj[()]
                    print(f'  data: {data}')
                elif obj.ndim >= 1:
                    sample = obj[0]
                    print(f'  sample[0]: {sample}')
            except Exception as e:
                print(f'  preview failed: {e}')

            print()

    print(f'=== {h5_path} ===')
    print('Top-level keys:', list(f.keys()))
    print('File attrs:', dict(f.attrs))
    print()

    f.visititems(show)
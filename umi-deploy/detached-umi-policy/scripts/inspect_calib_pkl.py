"""
Inspect hand-eye calibration pickle file format.
Dumps first few entries to verify tcp_pose representation (RotVec vs Euler).
"""
import sys
import os
import pickle
import numpy as np
import scipy.spatial.transform as st

def main():
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/yd/program/nyx/arx-difussion-deploy/umi-deploy/data_local/hand_eye_recalib_640/hand_eye_calib.pkl"
    
    print(f"=== Inspecting: {pkl_path} ===")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type: {type(data)}")
    print(f"Length: {len(data)}")
    
    if isinstance(data, list):
        for i, entry in enumerate(data[:5]):
            print(f"\n--- Entry {i} ---")
            if isinstance(entry, dict):
                for k, v in entry.items():
                    if k == 'img':
                        print(f"  {k}: ndarray shape={v.shape} dtype={v.dtype}")
                    elif isinstance(v, np.ndarray):
                        print(f"  {k}: ndarray shape={v.shape} dtype={v.dtype}")
                        print(f"       values: {v}")
                    else:
                        print(f"  {k}: {type(v).__name__} = {v}")
            else:
                print(f"  type: {type(entry)}")
        
        # Analyze tcp_pose format
        print("\n" + "="*60)
        print("=== TCP POSE FORMAT ANALYSIS ===")
        print("="*60)
        
        poses = []
        for entry in data:
            if isinstance(entry, dict) and 'tcp_pose' in entry:
                poses.append(entry['tcp_pose'])
        
        if not poses:
            print("No tcp_pose found in pkl!")
            return
        
        poses = np.array(poses)
        print(f"\nAll tcp_pose values ({len(poses)} samples):")
        print(f"  Position range:")
        print(f"    x: [{poses[:,0].min():.4f}, {poses[:,0].max():.4f}]")
        print(f"    y: [{poses[:,1].min():.4f}, {poses[:,1].max():.4f}]")
        print(f"    z: [{poses[:,2].min():.4f}, {poses[:,2].max():.4f}]")
        
        rot_part = poses[:, 3:]
        print(f"\n  Rotation part [3:6]:")
        print(f"    col3: [{rot_part[:,0].min():.4f}, {rot_part[:,0].max():.4f}]")
        print(f"    col4: [{rot_part[:,1].min():.4f}, {rot_part[:,1].max():.4f}]")
        print(f"    col5: [{rot_part[:,2].min():.4f}, {rot_part[:,2].max():.4f}]")
        
        # Test 1: Interpret as RotVec
        rotvec_norms = np.linalg.norm(rot_part, axis=1)
        print(f"\n  If RotVec:")
        print(f"    angle norms (rad): [{rotvec_norms.min():.4f}, {rotvec_norms.max():.4f}]")
        print(f"    angle norms (deg): [{np.degrees(rotvec_norms.min()):.2f}, {np.degrees(rotvec_norms.max()):.2f}]")
        
        # Test 2: Interpret as Euler XYZ
        print(f"\n  If Euler XYZ (rad):")
        for j in range(min(3, len(poses))):
            try:
                R_rv = st.Rotation.from_rotvec(rot_part[j])
                euler_from_rv = R_rv.as_euler('xyz', degrees=True)
                print(f"    sample {j}: rotvec={rot_part[j]} -> euler_xyz(deg)={euler_from_rv}")
            except:
                print(f"    sample {j}: failed to convert")
        
        # Check if values look like Euler angles (typically each component < pi)
        euler_like = np.all(np.abs(rot_part) < np.pi, axis=1)
        print(f"\n  All |components| < pi (euler-like): {euler_like.sum()}/{len(euler_like)}")
        
        # Check if the ee2tcp transform is baked in
        print(f"\n  First 3 tcp_pose samples (full):")
        for j in range(min(3, len(poses))):
            print(f"    [{j}]: pos={poses[j,:3]}, rot={poses[j,3:]}")
            
            # Show the 4x4 matrix if interpreted as rotvec
            try:
                R = st.Rotation.from_rotvec(poses[j, 3:]).as_matrix()
                print(f"         R (as rotvec):")
                for row in R:
                    print(f"           [{row[0]:+.4f} {row[1]:+.4f} {row[2]:+.4f}]")
                
                # Check Z-axis direction (should point roughly down for a gripper looking down)
                z_axis = R[:, 2]
                print(f"         Z-axis: [{z_axis[0]:+.4f} {z_axis[1]:+.4f} {z_axis[2]:+.4f}]")
            except:
                print(f"         Failed to interpret as rotvec")

    elif isinstance(data, dict):
        print("Dict keys:", list(data.keys()))
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}")
            else:
                print(f"  {k}: {type(v).__name__}")

if __name__ == '__main__':
    main()

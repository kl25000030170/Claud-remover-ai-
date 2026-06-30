import cv2
import numpy as np
import subprocess
import os
import sys

def main():
    checkpoint_path = "inpainter_checkpoint.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Creating dummy checkpoint at {checkpoint_path} for testing validation...")
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "lib")))
        import torch
        from reconstruct import PartialConvUNet
        model = PartialConvUNet()
        torch.save(model.state_dict(), checkpoint_path)

    print("Generating mock test images of various channel configurations...")
    
    # 1. 3-Channel BGR image with simulated cloud patch
    img_3ch = np.zeros((200, 200, 3), dtype=np.uint8)
    img_3ch[:, :] = [34, 139, 34] # Forest green
    # Add a white cloud circle in the center
    cv2.circle(img_3ch, (100, 100), 30, (240, 240, 240), -1)
    
    # 2. 1-Channel Grayscale image
    img_1ch = cv2.cvtColor(img_3ch, cv2.COLOR_BGR2GRAY)
    
    # 3. 4-Channel BGRA image
    img_4ch = cv2.cvtColor(img_3ch, cv2.COLOR_BGR2BGRA)
    img_4ch[:, :, 3] = 255 # Opaque alpha channel
    
    # Save the images
    os.makedirs("test_inputs", exist_ok=True)
    cv2.imwrite("test_inputs/img_3ch.png", img_3ch)
    cv2.imwrite("test_inputs/img_1ch.png", img_1ch)
    cv2.imwrite("test_inputs/img_4ch.png", img_4ch)
    
    python_bin = "./.venv/bin/python"
    script = "./src/lib/reconstruct.py"
    
    os.makedirs("test_outputs", exist_ok=True)
    
    test_cases = [
        ("3-Channel PNG", "test_inputs/img_3ch.png"),
        ("1-Channel Grayscale PNG", "test_inputs/img_1ch.png"),
        ("4-Channel BGRA PNG", "test_inputs/img_4ch.png")
    ]
    
    for label, path in test_cases:
        print(f"\n--- Testing {label} ({path}) ---")
        cmd = [python_bin, script, "--image", path, "--out_dir", "test_outputs"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        print("Exit Code:", res.returncode)
        if res.stderr:
            print("Stderr Log:")
            print(res.stderr.strip())
        if res.stdout:
            print("Stdout (JSON response excerpt):")
            try:
                data = json.loads(res.stdout.strip())
                print(f"  Cloud Percentage: {data.get('cloudPercentage')}%")
                print(f"  PSNR: {data.get('psnr')} dB")
                print(f"  SSIM: {data.get('ssim')}")
                print(f"  Land Use: {data.get('primaryLandUse')}")
                print(f"  Terrain Map Path exists: {os.path.exists(data.get('terrainMapPath')) if data.get('terrainMapPath') else False}")
                print(f"  Reconstructed Path exists: {os.path.exists(data.get('reconstPath'))}")
            except Exception as e:
                print("  Failed to parse stdout:", e)
                print(res.stdout[:200])

if __name__ == "__main__":
    import json
    main()

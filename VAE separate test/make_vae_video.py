#!/usr/bin/env python3
"""Render the exact 33-frame tensor the VAE receives (obs#1) to mp4 + gif.

Reconstructs vae_input as DreamZeroPipeline._encode_image builds it:
  stitched (1,352,640,3) uint8 -> /255 -> *2-1 (bf16) = frame0 in [-1,1];
  frames 1..32 = zeros (0.0 in [-1,1] = mid-gray in pixel space).
Then denormalizes ((x+1)/2*255) for display. Frame0 = real camera mosaic,
frames 1-32 = gray padding — exactly what the VAE convolves over.
"""
import numpy as np, cv2, sys
from pathlib import Path
from PIL import Image

NUM_FRAMES = 33
FPS = 5
npz = sys.argv[1]; outdir = Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)

stitched = np.load(npz)["images"]          # (1,352,640,3) uint8
if stitched.ndim == 4: stitched = stitched[0]
H, W = stitched.shape[:2]

# frame0 normalized to [-1,1] (bf16 round-trip via float32 is visually identical),
# padding frames = 0.0 in [-1,1].
frame0 = stitched.astype(np.float32)/255.0*2.0 - 1.0
vol = np.zeros((NUM_FRAMES, H, W, 3), dtype=np.float32)   # 0.0 == mid-gray
vol[0] = frame0

# denormalize for display: [-1,1] -> [0,1] -> uint8
disp = np.clip((vol + 1.0)/2.0, 0, 1)
disp = (disp*255.0).round().astype(np.uint8)              # (33,352,640,3) RGB

mp4 = outdir/"vae_input_33frames.mp4"
w = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))
for fr in disp: w.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
w.release()
print("SAVED_MP4", mp4, "size", mp4.stat().st_size)

gif = outdir/"vae_input_33frames.gif"
imgs = [Image.fromarray(fr) for fr in disp]
imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000/FPS), loop=0)
print("SAVED_GIF", gif, "size", gif.stat().st_size)

# also save frame0 alone (the only real content) as a png
png = outdir/"vae_input_frame0_stitched.png"
Image.fromarray(disp[0]).save(png)
print("SAVED_PNG", png)
print("SHAPE", disp.shape, "dtype", disp.dtype)

# Test data — DreamZero vLLM-Omni pipeline (by stage)

All test data used, organized by pipeline stage. Copied from `test_data/` and
`VAE separate test/`. Model: `GEAR-Dreams/DreamZero-DROID`.

## 1_huggingface_plain/
The plain inputs as downloaded / decoded from HuggingFace — before any encoding.
- `original_camera_mp4/` — 3 raw source camera clips (exterior_1, exterior_2, wrist)
- `original_camera_frames.npz` — 24 decoded RGB frames per view, `(24,180,320,3)` uint8
- `original_inputs.npz` — the single observation consumed: 3 first-frames + robot state
- `original_prompt.txt` — raw instruction prompt (text instruction)
- `templated_prompt.txt` — OXE_DROID templated prompt actually tokenized by UMT5

## 2_vae_input_3videos/
The stitched 3-view frame fed into the encoders (VAE input).
- `model_input_stitched.npz` — stitched 3-view frame, `(1,352,640,3)` uint8

## 3_vae_input_33frames/
The actual 33-frame video used for the isolated VAE encode calculation.
- `vae_input_33frames.mp4` / `.gif` — the 33-frame clip
- `vae_input_frame0_stitched.png` — first stitched frame

## 4_dit_input_encoded/
The encoded tensors that are passed into the DiT (encoder outputs).
- `text_encoder_outputs.pt` — UMT5 `prompt_embeds`, `negative_prompt_embeds`
- `vae_video_embeddings.pt` — VAE `image_latents`, `state_ys`, CLIP `state_clip_feas`
- `dit_inputs.pt` — complete set of tensors passed into the DiT
- `manifest.json` — shapes/dtypes of all embeddings + run results

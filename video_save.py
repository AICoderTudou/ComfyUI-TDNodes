import os
import sys
import json
import subprocess
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
import shutil
import logging

# Helper functions
def tensor_to_int(tensor, bits):
    tensor = tensor.cpu().numpy() * (2**bits-1) + 0.5
    return np.clip(tensor, 0, (2**bits-1))

def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)

def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)

def get_ffmpeg_path():
    path = shutil.which("ffmpeg")
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
        
    if path is None:
        # Try to look in common locations or environment variables if needed
        pass
    return path

class TD_VideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("INT", {"default": 8, "min": 1, "max": 120, "step": 1}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "TD_Video"}),
                "format": (["video/h264-mp4", "image/gif", "image/webp"],),
                "save_output": ("BOOLEAN", {"default": True}),
                "quality": ("INT", {"default": 85, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "AUDIO", "IMAGE")
    RETURN_NAMES = ("filenames", "audio", "images")
    OUTPUT_NODE = True
    FUNCTION = "combine_video"
    CATEGORY = "TDNodes"

    def combine_video(
        self,
        images,
        frame_rate: int,
        loop_count: int,
        filename_prefix="TD_Video",
        format="video/h264-mp4",
        save_output=True,
        quality=85,
        audio=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        # 1. Setup paths
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)

        # 2. Counter handling (simplified)
        counter = 0
        existing = [f for f in os.listdir(full_output_folder) if f.startswith(filename)]
        if existing:
            for f in existing:
                try:
                    # simplistic counter extraction
                    parts = f.split('_')
                    if len(parts) > 1:
                        num_part = parts[-1].split('.')[0]
                        if num_part.isdigit():
                            c = int(num_part)
                            if c > counter:
                                counter = c
                except:
                    pass
        counter += 1

        # 3. Determine Format and Extension
        format_type, ext_full = format.split("/")
        ext = ext_full.split("-")[-1] # h264-mp4 -> mp4
        
        output_file = f"{filename}_{counter:05}.{ext}"
        output_path = os.path.join(full_output_folder, output_file)

        ffmpeg_path = get_ffmpeg_path()
        if format_type == "video" and ffmpeg_path is None:
            raise ProcessLookupError("ffmpeg is required for video outputs but was not found.")

        # 4. Save Logic
        if format_type == "image":
            # Pillow handling for gif/webp
            pil_images = []
            for img in images:
                pil_images.append(Image.fromarray(tensor_to_bytes(img)))
            
            save_kwargs = {
                "save_all": True,
                "append_images": pil_images[1:],
                "duration": round(1000 / frame_rate),
                "loop": loop_count,
                "compress_level": 4,
            }
            if ext == "webp":
                save_kwargs["quality"] = quality
            
            pil_images[0].save(output_path, **save_kwargs)
        
        else:
            # ffmpeg handling
            # Ensure dimensions are even for libx264/yuv420p
            dim_alignment = 2
            first_image = images[0]
            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
                # Calculate padding
                to_pad = (
                    -first_image.shape[1] % dim_alignment,
                    -first_image.shape[0] % dim_alignment
                )
                padding = (
                    to_pad[0] // 2, to_pad[0] - to_pad[0] // 2,
                    to_pad[1] // 2, to_pad[1] - to_pad[1] // 2
                )
                
                padfunc = torch.nn.ReplicationPad2d(padding)
                
                def pad_image(image):
                    # HWC to CHW for padding
                    image_chw = image.permute(2, 0, 1)
                    padded = padfunc(image_chw.unsqueeze(0).to(dtype=torch.float32)).squeeze(0)
                    # Back to HWC
                    return padded.permute(1, 2, 0)
                
                # Apply padding to all images
                # Note: images might be a list or a tensor. If it's a tensor, we can pad it all at once if memory allows,
                # but map/iterator is safer for lists.
                if isinstance(images, torch.Tensor):
                    # images is [B, H, W, C]
                    # Permute to [B, C, H, W] for pad
                    images_perm = images.permute(0, 3, 1, 2)
                    padded_images = padfunc(images_perm)
                    images = padded_images.permute(0, 2, 3, 1)
                else:
                    images = [pad_image(img) for img in images]
                
                # Update dimensions
                dimensions = (images[0].shape[1], images[0].shape[0])
                print(f"Video dimensions padded to: {dimensions[0]}x{dimensions[1]}")
            else:
                dimensions = (images[0].shape[1], images[0].shape[0])
            
            # Basic ffmpeg args
            args = [
                ffmpeg_path,
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{dimensions[0]}x{dimensions[1]}",
                "-pix_fmt", "rgb24",
                "-r", str(frame_rate),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", str(max(0, 35 - (quality // 3))), # approximate quality mapping
                "-preset", "medium",
            ]
            
            # Keep a copy for writing to ffmpeg before we potentially modify 'images' for return
            ffmpeg_images = images

            if loop_count > 0:
                loop_args = ["-vf", f"loop=loop={loop_count}:size={len(images)}:start=0"]
                args += loop_args
                
                # Update return images to reflect the looped video
                if isinstance(images, list):
                    images = images * (loop_count + 1)
                elif isinstance(images, torch.Tensor):
                    images = torch.cat([images] * (loop_count + 1), dim=0)

            args.append(output_path)

            env = os.environ.copy()
            process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            
            # Write frames
            for img in ffmpeg_images:
                # Ensure image is bytes
                img_bytes = tensor_to_bytes(img).tobytes()
                try:
                    process.stdin.write(img_bytes)
                except BrokenPipeError:
                    break
            
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise Exception(f"ffmpeg failed: {stderr.decode()}")

        # 5. Audio Muxing
        final_output_path = output_path
        if audio is not None and ffmpeg_path is not None:
            # audio is typically a dict with 'waveform' and 'sample_rate'
            waveform = audio.get('waveform')
            sample_rate = audio.get('sample_rate')
            
            if waveform is not None:
                # Create a temporary audio file or pipe it
                # We'll re-mux using ffmpeg
                output_with_audio = f"{filename}_{counter:05}_audio.{ext}"
                output_with_audio_path = os.path.join(full_output_folder, output_with_audio)
                
                # We need to write audio to a temp file or pipe it. Piping is cleaner.
                # However, complex muxing might be easier with file inputs.
                # Let's try to just use the existing video file and add audio.
                
                # Audio data preparation
                channels = waveform.size(1)
                audio_bytes = waveform.squeeze(0).transpose(0,1).numpy().tobytes()
                
                mux_args = [
                    ffmpeg_path,
                    "-y",
                    "-i", output_path,
                    "-f", "f32le",
                    "-ar", str(sample_rate),
                    "-ac", str(channels),
                    "-i", "-", # audio from stdin
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    output_with_audio_path
                ]
                
                process_mux = subprocess.Popen(
                    mux_args,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env
                )
                
                stdout_mux, stderr_mux = process_mux.communicate(input=audio_bytes)
                
                if process_mux.returncode == 0:
                    final_output_path = output_with_audio_path
                    # Optionally delete the video-only file
                    # os.remove(output_path) 
                else:
                    print(f"Audio muxing failed: {stderr_mux.decode()}")

        # 6. Return
        return {"ui": {"images": [{"filename": os.path.basename(final_output_path), "subfolder": subfolder, "type": "output" if save_output else "temp"}]}, 
                "result": (os.path.basename(final_output_path), audio, images)}


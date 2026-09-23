# engine/enhancer/video_upscaler.py
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union
import cv2

from renderer.video_processor import merge_audio

ENGINE_DIR = Path(__file__).resolve().parent.parent
REALESRGAN_DIR = ENGINE_DIR / "tools" / "realesrgan"
REALESRGAN_BIN = REALESRGAN_DIR / "realesrgan-ncnn-vulkan"
REALESRGAN_MODELS = REALESRGAN_DIR / "models"


class RealESRGANEnhancer:
    """
    Pluggable / Detachable Super-Resolution Video Enhancer using Real-ESRGAN & Broadcast Filters.
    
    Can be easily toggled on/off (`enabled=True/False` or `.detach()`).
    When detached or disabled, processing requests cleanly pass through without error.
    """

    def __init__(
        self,
        enabled: bool = True,
        scale: int = 2,
        model_name: str = "realesrgan-x4plus",
        gpu_id: str = "auto",
        binary_path: Optional[Union[str, Path]] = None,
        models_path: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            enabled: If False, the enhancer is detached/bypassed.
            scale: Upscaling factor (2, 3, or 4).
            model_name: Pretrained model ('realesrgan-x4plus', 'realesr-animevideov3', etc.).
            gpu_id: GPU device ID ('auto', '0', '1', etc.).
            binary_path: Custom path to `realesrgan-ncnn-vulkan` binary.
            models_path: Custom path to Real-ESRGAN models folder.
        """
        self.enabled = enabled
        self.scale = scale
        self.model_name = model_name
        self.gpu_id = str(gpu_id)
        self.bin_path = Path(binary_path) if binary_path else REALESRGAN_BIN
        self.models_path = Path(models_path) if models_path else REALESRGAN_MODELS

    def attach(self) -> "RealESRGANEnhancer":
        """Enables the upscaler in the pipeline."""
        self.enabled = True
        return self

    def detach(self) -> "RealESRGANEnhancer":
        """Detaches (bypasses) the upscaler in the pipeline."""
        self.enabled = False
        return self

    def is_available(self) -> bool:
        """Returns True if the Real-ESRGAN binary and models are installed."""
        return self.bin_path.is_file() and os.access(self.bin_path, os.X_OK) and self.models_path.is_dir()

    def upscale_video(
        self,
        input_video_path: Union[str, Path],
        output_video_path: Union[str, Path],
        source_audio_path: Optional[Union[str, Path]] = None,
        scale: Optional[int] = None,
        model_name: Optional[str] = None,
        preserve_audio: bool = True,
        verbose: bool = False,
    ) -> str:
        """
        Upscales an entire video using Real-ESRGAN with original audio retention.
        If `self.enabled` is False, bypasses AI upscaling and copies/remuxes directly.

        Args:
            input_video_path: Path to the processed video frames to upscale.
            output_video_path: Path for the resulting high-res video.
            source_audio_path: Path to the original input video containing the audio track.
            scale: Override scale factor (default: self.scale).
            model_name: Override model name (default: self.model_name).
            preserve_audio: Whether to merge audio from input to output.
            verbose: Print verbose output.

        Returns:
            str: Path to the finalized video file.
        """
        input_video_path = Path(input_video_path).resolve()
        output_video_path = Path(output_video_path).resolve()
        audio_source = Path(source_audio_path).resolve() if source_audio_path else input_video_path
        scale = scale or self.scale
        model_name = model_name or self.model_name

        if not input_video_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # 1. DETACHED / BYPASSED MODE
        # -------------------------------------------------------------
        if not self.enabled:
            if verbose:
                print("[RealESRGAN] Enhancer is detached/disabled. Passing video through...")
            if preserve_audio and audio_source.exists():
                return merge_audio(input_video_path, audio_source, output_video_path)
            else:
                shutil.copy2(input_video_path, output_video_path)
                return str(output_video_path)

        # Check binary availability
        if not self.is_available():
            print(f"[WARNING] Real-ESRGAN binary not found at '{self.bin_path}'. Bypassing AI upscaling.")
            if preserve_audio and audio_source.exists():
                return merge_audio(input_video_path, audio_source, output_video_path)
            else:
                shutil.copy2(input_video_path, output_video_path)
                return str(output_video_path)

        # -------------------------------------------------------------
        # 2. ACTIVE REAL-ESRGAN UPSCALE PIPELINE
        # -------------------------------------------------------------
        cap = cv2.VideoCapture(str(input_video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        print(f"[RealESRGAN] Upscaling '{input_video_path.name}' (Scale: {scale}x, Model: {model_name}, FPS: {fps:.2f})...")

        with tempfile.TemporaryDirectory(prefix="realesrgan_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            frames_in_dir = tmp_path / "raw_frames"
            frames_out_dir = tmp_path / "upscaled_frames"
            frames_in_dir.mkdir()
            frames_out_dir.mkdir()

            # Step A: Extract lossless PNG frames
            print("  -> Extracting lossless frames...")
            extract_cmd = [
                "ffmpeg", "-y",
                "-i", str(input_video_path),
                str(frames_in_dir / "frame_%08d.png")
            ]
            subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

            # Step B: Run Real-ESRGAN multi-threaded inference on frames
            print(f"  -> Running GPU super-resolution inference with {model_name}...")
            realesrgan_cmd = [
                str(self.bin_path),
                "-i", str(frames_in_dir),
                "-o", str(frames_out_dir),
                "-s", str(scale),
                "-m", str(self.models_path),
                "-n", str(model_name),
                "-g", self.gpu_id,
                "-f", "png"
            ]
            if verbose:
                realesrgan_cmd.append("-v")

            res = subprocess.run(realesrgan_cmd, capture_output=not verbose, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Real-ESRGAN failed with error:\n{res.stderr}")

            # Step C: Re-encode upscaled frames into high-bitrate video
            print("  -> Re-encoding high-resolution video (CRF 16)...")
            temp_upscaled_video = tmp_path / "upscaled_no_audio.mp4"
            encode_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(frames_out_dir / "frame_%08d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "16",
                "-preset", "medium",
                str(temp_upscaled_video)
            ]
            subprocess.run(encode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

            # Step D: Preserve original audio track from audio_source
            if preserve_audio and audio_source.exists():
                print(f"  -> Merging audio track from '{audio_source.name}'...")
                merge_audio(
                    video_no_audio=temp_upscaled_video,
                    source_with_audio=audio_source,
                    output_path=output_video_path,
                    clean_temp=False
                )
            else:
                shutil.copy2(temp_upscaled_video, output_video_path)

        print(f"[RealESRGAN] Complete! Upscaled output saved to: {output_video_path}")
        return str(output_video_path)


def enhance_broadcast_4k(
    input_video_path: Union[str, Path],
    output_video_path: Union[str, Path],
    source_audio_path: Optional[Union[str, Path]] = None,
    target_width: int = 3840,
    target_height: int = 2160,
    sharpness: float = 0.5,
) -> str:
    """
    Fast, broadcast-grade 4K enhancement using Lanczos super-sampling,
    AMD Contrast Adaptive Sharpening (CAS), and high-bitrate H.264 encoding.
    Executes in seconds while preserving full audio.
    """
    input_video_path = Path(input_video_path).resolve()
    output_video_path = Path(output_video_path).resolve()
    audio_source = Path(source_audio_path).resolve() if source_audio_path else input_video_path

    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Broadcast 4K] Enhancing '{input_video_path.name}' -> {target_width}x{target_height} with Adaptive Sharpening...")

    # Filter graph: Lanczos scaling to 4K + CAS adaptive edge sharpening + unsharp contrast
    filter_chain = f"scale={target_width}:{target_height}:flags=lanczos,cas={sharpness}"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video_path),
        "-i", str(audio_source),
        "-vf", filter_chain,
        "-c:v", "libx264",
        "-crf", "16",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "320k",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        str(output_video_path)
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    print(f"[Broadcast 4K] Complete! Output saved to: {output_video_path}")
    return str(output_video_path)

# engine/renderer/video_processor.py
import os
import shutil
import subprocess
from pathlib import Path
from typing import Union
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FLAGS_DIR = Path(__file__).resolve().parent.parent / "assets" / "flags"


def merge_audio(
    video_no_audio: Union[str, Path],
    source_with_audio: Union[str, Path],
    output_path: Union[str, Path],
    clean_temp: bool = False,
    verbose: bool = False,
) -> str:
    """
    Merges the video stream from `video_no_audio` with the audio stream from `source_with_audio`
    into `output_path` using FFmpeg stream copy (lossless and near-instant).
    """
    video_no_audio = Path(video_no_audio).resolve()
    source_with_audio = Path(source_with_audio).resolve()
    output_path = Path(output_path).resolve()

    if not video_no_audio.exists():
        raise FileNotFoundError(f"Processed video not found: {video_no_audio}")
    if not source_with_audio.exists():
        raise FileNotFoundError(f"Source audio video not found: {source_with_audio}")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        if video_no_audio != output_path:
            shutil.copy2(video_no_audio, output_path)
        print("[WARNING] ffmpeg executable not found in PATH. Video saved without audio.")
        return str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(video_no_audio),
        "-i", str(source_with_audio),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        str(output_path)
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.PIPE if not verbose else None,
            check=True
        )
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg audio merge failed:\n{err_msg}") from e

    if clean_temp and video_no_audio != output_path and video_no_audio.exists():
        try:
            video_no_audio.unlink()
        except OSError:
            pass

    return str(output_path)


class BroadcastRenderer:
    # Expose merge_audio as a static method on the class for convenience
    merge_audio = staticmethod(merge_audio)

    def __init__(self, font_path: str = None, font_path_regular: str = None, scale: float = 1.0):
        """
        Initializes the rendering engine with modern compact sizing and flag support.
        Base scale=1.0 creates a sleek 236x60px broadcast overlay.
        """
        self.scale = scale
        self.bold_path = font_path or str(FONTS_DIR / "DMSans-Bold.ttf")
        self.regular_path = font_path_regular or str(FONTS_DIR / "DMSans-Regular.ttf")
        self._loaded_scale = None
        self._load_fonts(self.scale)
        self._flag_cache = {}

    def _load_fonts(self, scale: float):
        self._loaded_scale = scale
        try:
            self.font_name = ImageFont.truetype(self.bold_path, max(11, int(26 * scale)))
            self.font_pts = ImageFont.truetype(self.bold_path, max(13, int(30 * scale)))
            self.font_games = ImageFont.truetype(self.bold_path, max(10, int(23 * scale)))
        except OSError:
            self.font_name = ImageFont.load_default(size=max(11, int(26 * scale)))
            self.font_pts = ImageFont.load_default(size=max(13, int(30 * scale)))
            self.font_games = ImageFont.load_default(size=max(10, int(23 * scale)))

    def get_flag_image(self, country_code: str, target_w: int, target_h: int) -> Image.Image:
        """
        Loads and returns a national flag icon resized to target dimensions.
        Uses cached local assets in assets/flags/, downloads on-demand if needed,
        or generates a clean 2-letter national code badge fallback.
        """
        country_code = (country_code or "NG").strip().upper()
        cache_key = (country_code, target_w, target_h)
        if cache_key in self._flag_cache:
            return self._flag_cache[cache_key]

        FLAGS_DIR.mkdir(parents=True, exist_ok=True)
        flag_file = FLAGS_DIR / f"{country_code.lower()}.png"

        # If not present locally, attempt to download
        if not flag_file.exists():
            import urllib.request
            url = f"https://flagcdn.com/w40/{country_code.lower()}.png"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=3) as resp:
                    flag_file.write_bytes(resp.read())
            except Exception:
                pass

        if flag_file.exists():
            try:
                raw_flag = Image.open(flag_file).convert("RGBA")
                flag_resized = raw_flag.resize((target_w, target_h), Image.LANCZOS)
                self._flag_cache[cache_key] = flag_resized
                return flag_resized
            except Exception:
                pass

        # Stylized 2-letter fallback badge
        badge = Image.new("RGBA", (target_w, target_h), (40, 40, 50, 255))
        bdraw = ImageDraw.Draw(badge)
        bdraw.rounded_rectangle([0, 0, target_w, target_h], radius=2, fill=(45, 45, 55, 255), outline=(120, 120, 140, 255))
        bfont = ImageFont.load_default(size=max(8, target_h // 2))
        bbox = bdraw.textbbox((0, 0), country_code[:2], font=bfont)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bdraw.text(((target_w - bw) // 2, (target_h - bh) // 2), country_code[:2], font=bfont, fill=(255, 255, 255, 255))
        self._flag_cache[cache_key] = badge
        return badge

    @staticmethod
    def _draw_centered(draw, box, text, font, fill):
        """Draws `text` centered (both axes) inside box=(x0, y0, x1, y1)."""
        x0, y0, x1, y1 = box
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = x0 + ((x1 - x0) - w) / 2 - bbox[0]
        y = y0 + ((y1 - y0) - h) / 2 - bbox[1]
        draw.text((x, y), text, font=font, fill=fill)

    @staticmethod
    def _draw_left_vcenter(draw, box, text, font, fill):
        """Left-aligned, vertically centered text inside box=(x0, y0, x1, y1)."""
        x0, y0, x1, y1 = box
        bbox = draw.textbbox((0, 0), text, font=font)
        h = bbox[3] - bbox[1]
        y = y0 + ((y1 - y0) - h) / 2 - bbox[1]
        draw.text((x0, y), text, font=font, fill=fill)

    def draw_scorecard(
        self,
        frame: np.ndarray,
        p1_name: str = "Player 1",
        p2_name: str = "Player 2",
        p1_pts: int = 0,
        p2_pts: int = 0,
        p1_games: int = 0,
        p2_games: int = 0,
        server: int = 1,
        p1_country: str = "NG",
        p2_country: str = "NG",
        scale: float = None,
    ) -> np.ndarray:
        """
        Burns a WTT-style modern compact broadcast overlay with national flags on the frame.
        """
        s = scale if scale is not None else self.scale
        if s != self._loaded_scale:
            self._load_fonts(s)

        h_frame, w_frame, _ = frame.shape

        # ---- Sleek compact dimensions ----
        SS = 3  # supersample factor for smooth anti-aliased rendering
        row_h = int(30 * s)
        board_w = int(236 * s)
        board_h = row_h * 2
        radius = max(3, int(5 * s))
        margin = max(10, int(16 * s))

        # Column breakpoints
        serve_col_w = max(6, int(8 * s))
        flag_x0 = serve_col_w + max(4, int(5 * s))
        flag_w = int(20 * s)
        flag_h = int(14 * s)
        name_x0 = flag_x0 + flag_w + max(4, int(6 * s))

        pts_col_w = int(42 * s)
        games_col_w = int(36 * s)
        pts_x0 = board_w - games_col_w - pts_col_w
        games_x0 = board_w - games_col_w

        # ---- Palette ----
        bg_dark = (20, 20, 24, 235)
        bg_dark_alt = (26, 26, 31, 235)
        purple_accent = (168, 0, 255, 255)
        purple_dim = (90, 20, 130, 255)
        games_bg = (34, 34, 40, 240)
        text_white = (255, 255, 255, 255)
        text_muted = (160, 160, 168, 255)
        divider = (48, 48, 55, 255)
        border_col = (255, 255, 255, 28)

        canvas_w = (board_w + margin * 2) * SS
        canvas_h = (board_h + margin * 2) * SS
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        ox, oy = margin * SS, margin * SS

        # ---- Soft drop shadow ----
        shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow)
        sdraw.rounded_rectangle(
            [ox, oy + int(2 * s) * SS, ox + board_w * SS, oy + board_h * SS + int(2 * s) * SS],
            radius=radius * SS, fill=(0, 0, 0, 130),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(int(4 * s) * SS))
        canvas = Image.alpha_composite(canvas, shadow)

        # ---- Board base ----
        board_mask = Image.new("L", (canvas_w, canvas_h), 0)
        ImageDraw.Draw(board_mask).rounded_rectangle(
            [ox, oy, ox + board_w * SS, oy + board_h * SS], radius=radius * SS, fill=255
        )
        board_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        bl_draw = ImageDraw.Draw(board_layer)

        row_bgs = (bg_dark, bg_dark_alt)
        row_names = (p1_name, p2_name)
        row_countries = (p1_country, p2_country)
        row_pts = (p1_pts, p2_pts)
        row_games = (p1_games, p2_games)

        for i in range(2):
            ry0 = oy + i * row_h * SS
            ry1 = ry0 + row_h * SS

            # Row background
            bl_draw.rectangle([ox, ry0, ox + games_x0 * SS, ry1], fill=row_bgs[i])
            # Points block
            bl_draw.rectangle([ox + pts_x0 * SS, ry0, ox + games_x0 * SS, ry1], fill=purple_accent)
            # Games/sets block
            bl_draw.rectangle([ox + games_x0 * SS, ry0, ox + board_w * SS, ry1], fill=games_bg)

            # Serve indicator
            is_serving = (server == 1 and i == 0) or (server == 2 and i == 1)
            strip_fill = purple_accent if is_serving else purple_dim
            bl_draw.rectangle([ox, ry0, ox + serve_col_w * SS, ry1], fill=strip_fill)

            # National Flag icon
            flag_img = self.get_flag_image(row_countries[i], flag_w * SS, flag_h * SS)
            fy = ry0 + (row_h * SS - flag_h * SS) // 2
            fx = ox + flag_x0 * SS
            board_layer.paste(flag_img, (fx, fy), mask=flag_img if flag_img.mode == "RGBA" else None)
            # Subtle border around flag for contrast
            bl_draw.rectangle([fx, fy, fx + flag_w * SS, fy + flag_h * SS], outline=(255, 255, 255, 45), width=1)

            # Player name
            self._draw_left_vcenter(
                bl_draw,
                (ox + name_x0 * SS, ry0, ox + (pts_x0 - int(4 * s)) * SS, ry1),
                row_names[i].upper(),
                self.font_name,
                text_white,
            )

            # Points value
            self._draw_centered(
                bl_draw,
                (ox + pts_x0 * SS, ry0, ox + games_x0 * SS, ry1),
                str(row_pts[i]),
                self.font_pts,
                text_white,
            )
            # Games value
            self._draw_centered(
                bl_draw,
                (ox + games_x0 * SS, ry0, ox + board_w * SS, ry1),
                str(row_games[i]),
                self.font_games,
                text_muted,
            )

        # Divider line between the two rows
        bl_draw.line(
            [ox, oy + row_h * SS, ox + board_w * SS, oy + row_h * SS],
            fill=divider, width=1 * SS,
        )

        transparent = Image.new("RGBA", board_layer.size, (0, 0, 0, 0))
        board_layer = Image.composite(board_layer, transparent, board_mask)
        canvas = Image.alpha_composite(canvas, board_layer)

        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(
            [ox, oy, ox + board_w * SS, oy + board_h * SS],
            radius=radius * SS, outline=border_col, width=1 * SS,
        )

        # ---- Downsample for anti-aliasing ----
        final_w, final_h = board_w + margin * 2, board_h + margin * 2
        overlay = canvas.resize((final_w, final_h), Image.LANCZOS)

        # ---- Composite onto the frame ----
        pos_x, pos_y = 30 - margin, 25 - margin  # Sleek top-left position (30, 25)

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        frame_pil.paste(overlay, (pos_x, pos_y), mask=overlay)

        return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)


if __name__ == "__main__":
    # Quick manual preview
    renderer = BroadcastRenderer()
    frame = np.full((400, 700, 3), 200, dtype=np.uint8)
    for y in range(frame.shape[0]):
        shade = int(190 + 30 * np.sin(y / 60))
        frame[y, :, :] = shade
    out = renderer.draw_scorecard(
        frame, p1_name="David", p2_name="Chukwuka",
        p1_pts=1, p2_pts=0, p1_games=1, p2_games=0, server=2,
    )
    cv2.imwrite("/mnt/user-data/outputs/preview.png", out)
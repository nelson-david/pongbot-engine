# engine/renderer/video_processor.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


class BroadcastRenderer:
    def __init__(self, font_path: str = None, font_path_regular: str = None):
        """
        Initializes the rendering engine with custom font weights if available.
        Falls back gracefully to default PIL system fonts if truetype paths are missing.

        Fonts are loaded at 2x their display size because the scorecard is drawn
        on a supersampled canvas (see SS_FACTOR in draw_scorecard) and downsampled
        for anti-aliasing.
        """
        bold_path = font_path or "Helvetica-Bold.ttf"
        regular_path = font_path_regular or "Helvetica.ttf"
        try:
            self.font_name = ImageFont.truetype(bold_path, 40)
            self.font_pts = ImageFont.truetype(bold_path, 46)
            self.font_games = ImageFont.truetype(bold_path, 36)
        except OSError:
            # Fallback for systems without local Helvetica files installed
            self.font_name = ImageFont.load_default(size=40)
            self.font_pts = ImageFont.load_default(size=46)
            self.font_games = ImageFont.load_default(size=36)

    @staticmethod
    def _draw_centered(draw, box, text, font, fill):
        """Draws `text` centered (both axes) inside box=(x0, y0, x1, y1),
        using the glyph's actual ink bounding box so numerals sit dead-center
        regardless of font metrics/padding quirks."""
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
    ) -> np.ndarray:
        """
        Burns a WTT-style modern broadcast overlay on the frame.

        Args:
            frame: BGR image matrix from OpenCV.
            p1_name: Top/far player name.
            p2_name: Bottom/near player name.
            p1_pts / p2_pts: Current points in active rally.
            p1_games / p2_games: Total set/game score.
            server: 1 if P1 is serving, 2 if P2 is serving.
        """
        h_frame, w_frame, _ = frame.shape

        # ---- Layout (all in "display" pixels; SS_FACTOR supersamples for AA) ----
        SS = 3  # supersample factor for smooth edges/text
        row_h = 46
        board_w = 380
        board_h = row_h * 2
        radius = 8
        margin = 24  # extra canvas room so the drop shadow doesn't clip

        # Column breakpoints
        serve_col_w = 14      # slim accent strip that marks who's serving
        name_x0 = serve_col_w
        pts_col_w = 64
        games_col_w = 56
        pts_x0 = board_w - games_col_w - pts_col_w
        games_x0 = board_w - games_col_w

        # ---- Palette ----
        bg_dark = (20, 20, 24, 235)
        bg_dark_alt = (26, 26, 31, 235)     # very slightly lighter, for row 2 -> subtle separation
        purple_accent = (168, 0, 255, 255)
        purple_dim = (90, 20, 130, 255)     # accent strip when NOT serving
        games_bg = (34, 34, 40, 240)
        text_white = (255, 255, 255, 255)
        text_muted = (160, 160, 168, 255)
        divider = (48, 48, 55, 255)
        border_col = (255, 255, 255, 28)

        canvas_w = (board_w + margin * 2) * SS
        canvas_h = (board_h + margin * 2) * SS
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        ox, oy = margin * SS, margin * SS  # board origin on the supersampled canvas

        def R(*vals):
            return tuple(v * SS for v in vals)

        # ---- Soft drop shadow (blurred rounded rect behind the board) ----
        shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow)
        sdraw.rounded_rectangle(
            [ox, oy + 4 * SS, ox + board_w * SS, oy + board_h * SS + 4 * SS],
            radius=radius * SS, fill=(0, 0, 0, 130),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(6 * SS))
        canvas = Image.alpha_composite(canvas, shadow)
        draw = ImageDraw.Draw(canvas)

        # ---- Board base (rounded, clips everything drawn inside it) ----
        board_mask = Image.new("L", (canvas_w, canvas_h), 0)
        ImageDraw.Draw(board_mask).rounded_rectangle(
            [ox, oy, ox + board_w * SS, oy + board_h * SS], radius=radius * SS, fill=255
        )
        board_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        bl_draw = ImageDraw.Draw(board_layer)

        row_bgs = (bg_dark, bg_dark_alt)
        row_names = (p1_name, p2_name)
        row_pts = (p1_pts, p2_pts)
        row_games = (p1_games, p2_games)

        for i in range(2):
            ry0 = oy + i * row_h * SS
            ry1 = ry0 + row_h * SS

            # Row background (name area)
            bl_draw.rectangle([ox, ry0, ox + games_x0 * SS, ry1], fill=row_bgs[i])
            # Points block (accent color, always visually "live")
            bl_draw.rectangle([ox + pts_x0 * SS, ry0, ox + games_x0 * SS, ry1], fill=purple_accent)
            # Games/sets block
            bl_draw.rectangle([ox + games_x0 * SS, ry0, ox + board_w * SS, ry1], fill=games_bg)

            # Serve indicator: slim vertical accent strip on the far edge of the row
            is_serving = (server == 1 and i == 0) or (server == 2 and i == 1)
            strip_fill = purple_accent if is_serving else purple_dim
            bl_draw.rectangle([ox, ry0, ox + serve_col_w * SS, ry1], fill=strip_fill)

            # Player name: left-aligned, vertically centered in its column
            self._draw_left_vcenter(
                bl_draw,
                (ox + (name_x0 + 14) * SS, ry0, ox + (pts_x0 - 10) * SS, ry1),
                row_names[i].upper(),
                self.font_name,
                text_white,
            )

            # Points value, centered in its block
            self._draw_centered(
                bl_draw,
                (ox + pts_x0 * SS, ry0, ox + games_x0 * SS, ry1),
                str(row_pts[i]),
                self.font_pts,
                text_white,
            )
            # Games value, centered, muted (secondary info)
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

        # Clip the row content to the rounded board shape, then composite
        transparent = Image.new("RGBA", board_layer.size, (0, 0, 0, 0))
        board_layer = Image.composite(board_layer, transparent, board_mask)
        canvas = Image.alpha_composite(canvas, board_layer)

        # Hairline border for definition
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(
            [ox, oy, ox + board_w * SS, oy + board_h * SS],
            radius=radius * SS, outline=border_col, width=1 * SS,
        )

        # ---- Downsample for anti-aliasing ----
        final_w, final_h = board_w + margin * 2, board_h + margin * 2
        overlay = canvas.resize((final_w, final_h), Image.LANCZOS)

        # ---- Composite onto the frame ----
        pos_x, pos_y = 50 - margin, 50 - margin  # keep the board's visible top-left at (50, 50)

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
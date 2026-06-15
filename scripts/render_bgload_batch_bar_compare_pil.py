from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/workspace/Synapse/Synapse')
DEFAULT_CSV_PATH = ROOT / 'results/figures/bgload_batch_bar_compare.csv'
DEFAULT_PNG_PATH = ROOT / 'results/figures/bgload_batch_bar_compare.png'
DEFAULT_CLEAN_PATH = ROOT / 'results/figures/bgload_batch_bar_compare_clean.png'
FONT_REG = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

BG = (255, 255, 255)
AXIS = (34, 34, 34)
GRID = (210, 214, 220)
BLUE = (76, 120, 168)
BLUE_EDGE = (54, 91, 125)
ORANGE = (210, 122, 44)
ORANGE_EDGE = (158, 91, 32)
GREEN = (27, 127, 90)

W = 1910
H = 1208
LEFT = 110
RIGHT = 60
TOP = 90
BOTTOM = 110
PLOT_W = W - LEFT - RIGHT
PLOT_H = H - TOP - BOTTOM
Y_MAX = 58.0
Y_TICKS = [0, 10, 20, 30, 40, 50]
BAR_W = 160
GROUP_GAP = PLOT_W / 4.0


def load_rows(csv_path: Path) -> List[dict]:
    with csv_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['batch_size'] = int(row['batch_size'])
        row['tspipe_minutes'] = int(row['tspipe_total_seconds']) / 60.0
        row['failover_minutes'] = int(row['failover_total_seconds']) / 60.0
        row['improvement_pct'] = float(row['improvement_pct'])
    return rows


def y_to_px(value: float) -> float:
    return TOP + PLOT_H * (1.0 - value / Y_MAX)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_rotated_text(base: Image.Image, text: str, font: ImageFont.FreeTypeFont, fill: tuple[int, int, int], x: int, y: int) -> None:
    tmp = Image.new('RGBA', (220, 800), (255, 255, 255, 0))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    d.text((0, 0), text, font=font, fill=fill)
    tmp = tmp.crop((0, 0, tw, th)).rotate(90, expand=True)
    base.alpha_composite(tmp, (x, y))


def draw_dashed_hline(draw: ImageDraw.ImageDraw, y: int, x0: int, x1: int, color: tuple[int, int, int]) -> None:
    dash = 10
    gap = 10
    x = x0
    while x < x1:
        draw.line((x, y, min(x + dash, x1), y), fill=color, width=2)
        x += dash + gap


def render(csv_path: Path, png_path: Path, clean_path: Path) -> None:
    rows = load_rows(csv_path)
    img = Image.new('RGBA', (W, H), BG + (255,))
    draw = ImageDraw.Draw(img)

    title_font = ImageFont.truetype(FONT_REG, 36)
    axis_font = ImageFont.truetype(FONT_REG, 28)
    tick_font = ImageFont.truetype(FONT_REG, 24)
    label_font = ImageFont.truetype(FONT_REG, 22)
    delta_font = ImageFont.truetype(FONT_BOLD, 26)
    legend_font = ImageFont.truetype(FONT_REG, 24)

    for tick in Y_TICKS:
        y = int(round(y_to_px(tick)))
        draw_dashed_hline(draw, y, LEFT, W - RIGHT, GRID)
        tw, th = text_size(draw, str(tick), tick_font)
        draw.text((LEFT - 18 - tw, y - th / 2), str(tick), font=tick_font, fill=AXIS)

    draw.line((LEFT, TOP, LEFT, TOP + PLOT_H), fill=AXIS, width=2)
    draw.line((LEFT, TOP + PLOT_H, W - RIGHT, TOP + PLOT_H), fill=AXIS, width=2)

    title = 'End-to-End Completion Time Under GPU-3 Background Load'
    tw, th = text_size(draw, title, title_font)
    draw.text(((W - tw) / 2, 20), title, font=title_font, fill=AXIS)

    x_label = 'Batch size'
    tw, th = text_size(draw, x_label, axis_font)
    draw.text(((W - tw) / 2, H - 58), x_label, font=axis_font, fill=AXIS)
    draw_rotated_text(img, 'Completion time (minutes)', axis_font, AXIS, 10, TOP + (PLOT_H - 320) // 2)

    legend_x = W - RIGHT - 420
    legend_y = 126
    box_w = 44
    box_h = 20
    draw.rectangle((legend_x, legend_y, legend_x + box_w, legend_y + box_h), fill=BLUE, outline=BLUE_EDGE, width=2)
    draw.text((legend_x + 60, legend_y - 8), 'TSPipe baseline', font=legend_font, fill=AXIS)
    draw.rectangle((legend_x, legend_y + 46, legend_x + box_w, legend_y + 46 + box_h), fill=ORANGE, outline=ORANGE_EDGE, width=2)
    draw.text((legend_x + 60, legend_y + 38), 'Failover + REPLAN', font=legend_font, fill=AXIS)

    for idx, row in enumerate(rows):
        group_center = LEFT + GROUP_GAP * (idx + 0.5)
        tspipe_left = int(round(group_center - BAR_W))
        failover_left = int(round(group_center))
        tspipe_right = tspipe_left + BAR_W
        failover_right = failover_left + BAR_W

        tspipe_top = int(round(y_to_px(row['tspipe_minutes'])))
        failover_top = int(round(y_to_px(row['failover_minutes'])))
        baseline_y = int(round(y_to_px(0)))

        draw.rectangle((tspipe_left, tspipe_top, tspipe_right, baseline_y), fill=BLUE, outline=BLUE_EDGE, width=3)
        draw.rectangle((failover_left, failover_top, failover_right, baseline_y), fill=ORANGE, outline=ORANGE_EDGE, width=3)

        batch_label = str(row['batch_size'])
        tw, th = text_size(draw, batch_label, tick_font)
        draw.text((group_center - tw / 2, baseline_y + 12), batch_label, font=tick_font, fill=AXIS)

        tspipe_text = f"{row['tspipe_minutes']:.1f}m"
        failover_text = f"{row['failover_minutes']:.1f}m"
        delta_text = f"-{row['improvement_pct']:.1f}%"

        tw, th = text_size(draw, tspipe_text, label_font)
        draw.text((tspipe_left + BAR_W / 2 - tw / 2, tspipe_top - th - 6), tspipe_text, font=label_font, fill=BLUE_EDGE)

        failover_center_x = failover_left + BAR_W / 2

        tw2, th2 = text_size(draw, failover_text, label_font)
        orange_x = failover_center_x - tw2 / 2
        orange_y = failover_top - th2 - 6
        draw.text((orange_x, orange_y), failover_text, font=label_font, fill=ORANGE_EDGE)

        tw3, th3 = text_size(draw, delta_text, delta_font)
        delta_x = failover_center_x - tw3 / 2
        delta_y = orange_y - th3 - 14
        draw.text((delta_x, delta_y), delta_text, font=delta_font, fill=GREEN)

    rgb = img.convert('RGB')
    png_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(png_path)
    rgb.save(clean_path)
    print(png_path)
    print(clean_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Render the clean batch bar comparison PNG from a CSV.')
    parser.add_argument(
        '--csv-path',
        type=Path,
        default=DEFAULT_CSV_PATH,
        help='Input CSV produced by plot_bgload_batch_bar_comparison.py.',
    )
    parser.add_argument(
        '--png-path',
        type=Path,
        default=DEFAULT_PNG_PATH,
        help='Primary PNG output path.',
    )
    parser.add_argument(
        '--clean-path',
        type=Path,
        default=DEFAULT_CLEAN_PATH,
        help='Secondary clean PNG output path.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    render(args.csv_path, args.png_path, args.clean_path)

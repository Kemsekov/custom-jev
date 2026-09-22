"""Per-task artifact directories.

Every evaluation task owns one directory under `results/artifacts/<task>/`.
The directory is wiped at the start of a run, so re-running a task always
replaces stale outputs instead of accumulating them. Textual tasks write
JSON/JSONL/Markdown; visual tasks additionally write contact sheets with the
expected vs chosen answer burned in.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from .config import PROJECT_ROOT

ARTIFACTS_ROOT = PROJECT_ROOT / "results" / "artifacts"

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)


def load_font(size: int = 14):
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                from PIL import ImageFont

                return ImageFont.truetype(path, size)
            except OSError:
                continue
    from PIL import ImageFont

    return ImageFont.load_default()


class TaskArtifacts:
    """Artifact directory for one task, reset on creation."""

    def __init__(self, key: str, reset: bool = True):
        self.key = key
        self.dir = ARTIFACTS_ROOT / key
        if reset:
            self.reset()
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
        self._files: list[str] = []

    def reset(self) -> "TaskArtifacts":
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._files = []
        return self

    # ------------------------------------------------------------------
    def path(self, *parts: str) -> Path:
        target = self.dir.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _track(self, path: Path) -> Path:
        rel = str(path.relative_to(self.dir))
        if rel not in self._files:
            self._files.append(rel)
        return path

    def write_json(self, name: str, data: Any) -> Path:
        path = self.path(name)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        return self._track(path)

    def write_jsonl(self, name: str, rows: Iterable[dict]) -> Path:
        path = self.path(name)
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return self._track(path)

    def write_text(self, name: str, text: str) -> Path:
        path = self.path(name)
        path.write_text(text)
        return self._track(path)

    def save_image(self, name: str, image: Image.Image) -> Path:
        path = self.path(name)
        image.save(path)
        return self._track(path)

    def summary(self, **meta: Any) -> Path:
        payload = {
            "task": self.key,
            "dir": str(self.dir),
            "files": sorted(self._files),
            **meta,
        }
        return self.write_json("summary.json", payload)


def contact_sheet(
    entries: list[dict],
    columns: int = 3,
    cell: int = 176,
    caption_lines: int = 2,
) -> Image.Image:
    """Grid of thumbnails with captions, used as visual decision evidence.

    entries: [{"image": path|PIL.Image, "title": str, "subtitle": str, "ok": bool}]
    """
    if not entries:
        return Image.new("RGB", (cell, cell), (255, 255, 255))
    rows = (len(entries) + columns - 1) // columns
    caption_h = 18 * caption_lines + 8
    pad = 10
    title_h = 26
    width = columns * (cell + pad) + pad
    height = rows * (cell + caption_h + pad + title_h) + pad
    sheet = Image.new("RGB", (width, height), (245, 245, 248))
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(14)
    small_font = load_font(12)

    for i, entry in enumerate(entries):
        col, row = i % columns, i // columns
        x = pad + col * (cell + pad)
        y = pad + row * (cell + caption_h + pad + title_h)
        draw.text((x, y), entry.get("title", ""), fill=(20, 20, 20), font=title_font)
        img = entry["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB")
        img.thumbnail((cell, cell))
        frame = Image.new("RGB", (cell, cell), (255, 255, 255))
        frame.paste(img, ((cell - img.width) // 2, (cell - img.height) // 2))
        sheet.paste(frame, (x, y + title_h))
        color = (0, 130, 0) if entry.get("ok", True) else (190, 30, 30)
        subtitle = entry.get("subtitle", "")
        draw.text(
            (x, y + title_h + cell + 2), subtitle, fill=color, font=small_font
        )
    return sheet


def artifacts_root() -> Path:
    return ARTIFACTS_ROOT

"""Maze navigation as a stream of JEV decisions.

A perfect maze is rendered as an image (black walls, red agent, green exit).
At every step the model gets only the picture plus a typed criterion and must
choose among the currently valid moves. The reference next move is the BFS
shortest-path direction, so both task success and per-decision accuracy can be
scored.
"""

from __future__ import annotations

import random
import shutil
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ..config import PROJECT_ROOT, Config

N, E, S, W = 1, 2, 4, 8
DIRS = {
    "up": (0, -1, N, S),
    "down": (0, 1, S, N),
    "left": (-1, 0, W, E),
    "right": (1, 0, E, W),
}


def generate_maze(width: int, height: int, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    walls = [[N | E | S | W for _ in range(width)] for _ in range(height)]
    visited = [[False] * width for _ in range(height)]
    stack = [(0, 0)]
    visited[0][0] = True
    while stack:
        x, y = stack[-1]
        candidates = []
        for name, (dx, dy, wall, opposite) in DIRS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                candidates.append((nx, ny, wall, opposite))
        if not candidates:
            stack.pop()
            continue
        nx, ny, wall, opposite = rng.choice(candidates)
        walls[y][x] &= ~wall
        walls[ny][nx] &= ~opposite
        visited[ny][nx] = True
        stack.append((nx, ny))
    return walls


def valid_moves(walls: list[list[int]], pos: tuple[int, int]) -> list[str]:
    x, y = pos
    moves = []
    for name, (dx, dy, wall, _) in DIRS.items():
        if not (walls[y][x] & wall):
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(walls[0]) and 0 <= ny < len(walls):
                moves.append(name)
    return moves


def move(pos: tuple[int, int], direction: str) -> tuple[int, int]:
    dx, dy, _, _ = DIRS[direction]
    return pos[0] + dx, pos[1] + dy


def bfs_distances(walls: list[list[int]], target: tuple[int, int]) -> dict[tuple[int, int], int]:
    width, height = len(walls[0]), len(walls)
    dist = {target: 0}
    queue = deque([target])
    while queue:
        pos = queue.popleft()
        for direction in valid_moves(walls, pos):
            nxt = move(pos, direction)
            if nxt not in dist:
                dist[nxt] = dist[pos] + 1
                queue.append(nxt)
    return dist


def optimal_direction(walls: list[list[int]], pos: tuple[int, int], target: tuple[int, int]) -> str | None:
    dist = bfs_distances(walls, target)
    best = None
    best_d = dist.get(pos, 10**9)
    for direction in valid_moves(walls, pos):
        d = dist.get(move(pos, direction), 10**9)
        if d < best_d:
            best_d = d
            best = direction
    return best


def render_maze(
    walls: list[list[int]],
    agent: tuple[int, int],
    exit_pos: tuple[int, int],
    cell_px: int = 96,
    wall_px: int = 6,
    path: list[tuple[int, int]] | None = None,
) -> Image.Image:
    width, height = len(walls[0]), len(walls)
    w_px = width * cell_px + wall_px
    h_px = height * cell_px + wall_px
    img = Image.new("RGB", (w_px, h_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    def grid(coord: int) -> int:
        """Center of the grid line preceding cell `coord`."""
        return coord * cell_px + wall_px // 2

    def cell_origin(pos):
        return (grid(pos[0]) + cell_px // 2, grid(pos[1]) + cell_px // 2)

    # exit cell background
    ex, ey = exit_pos
    x0, y0 = grid(ex), grid(ey)
    draw.rectangle(
        [x0, y0, x0 + cell_px, y0 + cell_px],
        fill=(190, 240, 190),
        outline=(0, 120, 0),
        width=wall_px // 2,
    )

    # trail of already visited cells (drawn under the walls)
    if path:
        centers = [
            (grid(p[0]) + cell_px // 2, grid(p[1]) + cell_px // 2) for p in path
        ]
        if len(centers) > 1:
            draw.line(centers, fill=(70, 130, 230), width=max(3, cell_px // 5), joint="curve")
        dot = max(2, cell_px // 12)
        for cx, cy in centers:
            draw.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill=(40, 100, 210))

    for y in range(height):
        for x in range(width):
            ox, oy = cell_origin((x, y))
            half = cell_px // 2
            if walls[y][x] & N:
                draw.line([(ox - half, oy - half), (ox + half, oy - half)], fill=(0, 0, 0), width=wall_px)
            if walls[y][x] & S:
                draw.line([(ox - half, oy + half), (ox + half, oy + half)], fill=(0, 0, 0), width=wall_px)
            if walls[y][x] & W:
                draw.line([(ox - half, oy - half), (ox - half, oy + half)], fill=(0, 0, 0), width=wall_px)
            if walls[y][x] & E:
                draw.line([(ox + half, oy - half), (ox + half, oy + half)], fill=(0, 0, 0), width=wall_px)

    ax, ay = agent
    cx, cy = grid(ax) + cell_px // 2, grid(ay) + cell_px // 2
    r = max(4, int(cell_px * 0.28))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 30, 30), outline=(120, 0, 0), width=2)
    return img


def build_state(
    pos: tuple[int, int],
    exit_pos: tuple[int, int],
    history: list[str],
    path: list[tuple[int, int]] | None = None,
    width: int = 0,
    height: int = 0,
) -> dict:
    """Explicit prompt-side description of everything visible in the render."""
    ordered: list[list[int]] = [[c[0] + 1, c[1] + 1] for c in (path or [pos])]
    visited: list[list[int]] = []
    seen = set()
    for cell in path or [pos]:
        if cell not in seen:
            seen.add(cell)
            visited.append([cell[0] + 1, cell[1] + 1])
    return {
        "maze": {
            "columns": width or None,
            "rows": height or None,
            "walls_look_like": (
                "thick black lines between cells; the outer border of the image "
                "is a wall too, so the agent can never leave the grid"
            ),
        },
        "current_position": {
            "looks_like": "a red circle with a dark red outline",
            "column": pos[0] + 1,
            "row": pos[1] + 1,
            "note": "column 1 is the left edge, row 1 is the top edge of the image",
        },
        "goal": {
            "looks_like": "a light green square with a dark green border",
            "column": exit_pos[0] + 1,
            "row": exit_pos[1] + 1,
            "rule": "the task ends when the red circle reaches the green square",
        },
        "previous_path": {
            "looks_like": "a blue line with blue dots through every visited cell",
            "positions_in_order": ordered[-12:],
            "visited_cells": visited,
            "visited_count": len(visited),
            "hint": "do not walk back into cells that are already behind you unless they are the only way forward",
        },
        "directions": {
            "up": "toward the top of the image",
            "down": "toward the bottom of the image",
            "left": "toward the left edge of the image",
            "right": "toward the right edge of the image",
        },
        "moves_already_taken": history[-12:] if history else "none",
        "rule": "move exactly one cell per turn and never cross a black wall",
    }


def solve_maze(
    engine,
    opts: dict[str, Any],
    cfg: Config | None = None,
    save_dir: Path | None = None,
) -> dict:
    cfg = cfg or engine.cfg
    width = int(opts.get("width", 6))
    height = int(opts.get("height", 6))
    seed = int(opts.get("seed", 0))
    max_steps = int(opts.get("max_steps", 40))
    cell_px = int(opts.get("cell_px", 96))
    thinking = opts.get("thinking")
    save_images = bool(opts.get("save_images", True))
    # legal_moves_only=True removes wall perception from the decision (the
    # harness only offers open moves); False tests wall awareness directly.
    legal_only = bool(opts.get("legal_moves_only", True))

    walls = generate_maze(width, height, seed)
    start = (0, 0)
    exit_pos = (width - 1, height - 1)
    pos = start
    dist = bfs_distances(walls, exit_pos)

    if save_images:
        base = Path(save_dir) if save_dir else PROJECT_ROOT / "results" / "artifacts" / "maze"
        save_dir = base / f"seed-{seed}"
        shutil.rmtree(save_dir, ignore_errors=True)
        save_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict] = []
    history: list[str] = []
    path: list[tuple[int, int]] = [start]
    success = False
    for step in range(max_steps):
        open_moves = valid_moves(walls, pos)
        image = render_maze(walls, pos, exit_pos, cell_px, path=path)
        if legal_only and len(open_moves) == 1:
            forced = open_moves[0]
            reference = optimal_direction(walls, pos, exit_pos)
            if save_images:
                image.save(save_dir / f"step-{step:02d}.png")
            steps.append(
                {
                    "step": step,
                    "pos": list(pos),
                    "open_moves": open_moves,
                    "options": [forced],
                    "chosen": forced,
                    "reference": reference,
                    "valid_move": True,
                    "forced": True,
                    "match_reference": forced == reference,
                    "probabilities": {},
                    "confidence": 1.0,
                    "prompt_tokens": 0,
                    "prompt_ms": 0.0,
                }
            )
            history.append(forced)
            pos = move(pos, forced)
            path.append(pos)
            if pos == exit_pos:
                success = True
                break
            continue
        move_set = list(open_moves) if legal_only else list(DIRS)
        options = [
            {"id": m, "description": f"move {m} (toward the {m} of the image)"}
            for m in move_set
        ]
        result = engine.decide(
            question=(
                "The red circle is the agent, the green square is the exit, the "
                "blue line is the path already taken, and the black lines are "
                "walls. Which move should the agent take next to reach the "
                "exit? The move must stay inside the maze and must not cross a "
                "black wall."
            ),
            options=options,
            state=build_state(pos, exit_pos, history, path, width, height),
            image=image,
            thinking=thinking,
        )
        reference = optimal_direction(walls, pos, exit_pos)
        valid = result.chosen in open_moves
        if save_images:
            image.save(save_dir / f"step-{step:02d}.png")
        steps.append(
            {
                "step": step,
                "pos": list(pos),
                "open_moves": open_moves,
                "options": [o["id"] for o in options],
                "chosen": result.chosen,
                "reference": reference,
                "valid_move": valid,
                "match_reference": result.chosen == reference,
                "probabilities": {o.id: round(o.probability, 4) for o in result.options},
                "confidence": round(result.confidence, 4),
                "prompt_tokens": result.tokens_evaluated,
                "prompt_ms": result.timings.get("prompt_ms", 0.0),
            }
        )
        history.append(result.chosen)
        if valid:
            pos = move(pos, result.chosen)
            path.append(pos)
        if pos == exit_pos:
            success = True
            break

    final_image = render_maze(walls, pos, exit_pos, cell_px, path=path)
    if save_images:
        final_image.save(save_dir / "final.png")

    matched = sum(1 for s in steps if s["match_reference"])
    invalid = sum(1 for s in steps if not s["valid_move"])
    optimal_len = dist.get(start, 0)
    return {
        "success": success,
        "steps": len(steps),
        "optimal_steps": optimal_len,
        "invalid_moves": invalid,
        "step_optimality": matched / len(steps) if steps else 0.0,
        "decisions": steps,
        "start": list(start),
        "exit": list(exit_pos),
        "final_pos": list(pos),
        "seed": seed,
        "maze": {"width": width, "height": height},
        "images": str(save_dir) if save_images else None,
    }

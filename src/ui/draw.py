from pathlib import Path
from typing import Any, List, Tuple, Union

import pygame
from pgzero.rect import Rect

font = pygame.font.Font(None, 15)


def draw_rectangle_on_surface(
    surface: pygame.Surface,
    x: int,
    y: int,
    width: int,
    height: int,
    background_color: Union[str, Tuple[int, ...]] = "white",
    outline_color: Union[str, Tuple[int, ...]] = "black",
    text: str = None,
):
    # Define the rectangle (x, y, width, height)
    rect = Rect(
        x,
        y,
        width,
        height,
    )

    # Draw filled color
    pygame.draw.rect(surface, background_color, rect)

    # Draw outline color
    pygame.draw.rect(surface, outline_color, rect, width=2)

    # Draw text if provided
    if text:
        text_surface = font.render(text, True, "black")
        text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))

        surface.blit(text_surface, text_rect)


def draw_rectangle_on_screen(
    screen: Any,
    x: int,
    y: int,
    width: int,
    height: int,
    outline_color: Union[str, Tuple[int, ...]] = "cyan",
):
    screen.draw.rect(Rect((x, y), (width, height)), outline_color)


def draw_arrow_on_surface(
    surface: pygame.Surface,
    x: int,
    y: int,
    length: int = 50,
    color: Union[str, Tuple[int, ...]] = "black",
    direction: str = "right",
    shaft_width: int = 6,
    arrow_head_size: int = 6,
):

    shaft_adjustment = 3
    if direction == "right":
        # Shaft
        pygame.draw.line(
            surface, color, (x, y), (x + length - shaft_adjustment, y), shaft_width
        )
        # Arrowhead points (triangle)
        points = [
            (x + length, y),
            (x + length - arrow_head_size, y - arrow_head_size),
            (x + length - arrow_head_size, y + arrow_head_size),
        ]
        pygame.draw.polygon(surface, color, points)

    elif direction == "left":
        pygame.draw.line(
            surface, color, (x, y), (x - length + shaft_adjustment, y), shaft_width
        )
        points = [
            (x - length, y),
            (x - length + arrow_head_size, y - arrow_head_size),
            (x - length + arrow_head_size, y + arrow_head_size),
        ]
        pygame.draw.polygon(surface, color, points)

    elif direction == "up":
        pygame.draw.line(
            surface, color, (x, y), (x, y - length + shaft_adjustment), shaft_width
        )
        points = [
            (x, y - length),
            (x - arrow_head_size, y - length + arrow_head_size),
            (x + arrow_head_size, y - length + arrow_head_size),
        ]
        pygame.draw.polygon(surface, color, points)

    elif direction == "down":
        pygame.draw.line(
            surface, color, (x, y), (x, y + length - shaft_adjustment), shaft_width
        )
        points = [
            (x, y + length),
            (x - arrow_head_size, y + length - arrow_head_size),
            (x + arrow_head_size, y + length - arrow_head_size),
        ]
        pygame.draw.polygon(surface, color, points)


def draw_image_on_surface(surface: pygame.Surface, filepath: Path, x: int, y: int):
    image_surface = pygame.image.load(filepath).convert_alpha()
    surface.blit(
        image_surface,
        (
            x,
            y,
        ),
    )


def draw_text_on_screen(
    screen: Any,
    text: str,
    x_range: List[int],
    y_range: List[int],
    align: str = ("left", "top"),
    color: Union[str, Tuple[int, ...]] = "white",
    font=None,
    line_spacing: int = 5,
):
    # Split up lines by EOL symbol
    lines = text.split("\n")
    rendered_lines = [font.render(line, True, pygame.Color(color)) for line in lines]

    # Measure total height
    line_heights = [surface.get_height() for surface in rendered_lines]
    total_height = sum(line_heights) + line_spacing * (len(lines) - 1)

    # Vertical alignment
    y_start, y_end = y_range
    if align[1] == "top":
        y = y_start
    elif align[1] == "middle":
        y = y_start + (y_end - y_start - total_height) // 2
    elif align[1] == "bottom":
        y = y_end - total_height
    else:
        raise ValueError("Invalid vertical alignment: use 'top', 'middle', or 'bottom'")

    # Draw each line with horizontal alignment
    x_start, x_end = x_range
    for i, surface in enumerate(rendered_lines):
        line_width = surface.get_width()
        if align[0] == "left":
            x = x_start
        elif align[0] == "center":
            x = x_start + (x_end - x_start - line_width) // 2
        elif align[0] == "right":
            x = x_end - line_width
        else:
            raise ValueError(
                "Invalid horizontal alignment: use 'left', 'center', or 'right'"
            )

        screen.surface.blit(surface, (x, y))
        y += line_heights[i] + line_spacing

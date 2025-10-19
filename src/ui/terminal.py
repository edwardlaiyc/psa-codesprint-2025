from typing import Any, Dict

import pygame

from src.constant import CONSTANT
from src.ui.draw import (
    draw_arrow_on_surface,
    draw_image_on_surface,
    draw_rectangle_on_screen,
    draw_rectangle_on_surface,
    draw_text_on_screen,
)
from src.ui.job_progress import JobProgressDisplay


class TerminalDisplay:
    """
    Manages the graphical display of the terminal, including rendering the base surface
    and tracking job progress with a dedicated display component.

    Attributes
    ----------
    surface : pygame.Surface
        The rendered surface representing the terminal display.
    job_tracker : JobProgressDisplay
        An instance responsible for tracking and displaying job progress.
    tracker_font : pygame.font.Font
        The font used for displaying text within the job tracker.
    """

    def __init__(self):
        self.surface: pygame.Surface = self.__render()
        self.job_tracker = JobProgressDisplay()
        self.tracker_font = pygame.font.SysFont("Courier New", 17)

    def __render(self) -> pygame.Surface:
        surface = pygame.Surface(
            (CONSTANT.SCREEN.WIDTH_PX, CONSTANT.SCREEN.HEIGHT_PX), pygame.SRCALPHA
        )
        self.__draw_coordinates_map(surface)
        self.__draw_lanes_sectors(surface)
        self.__draw_quay_crane_sectors(surface)
        self.__draw_yard_sectors(surface)
        return surface

    def __draw_coordinates_map(self, surface: pygame.Surface):
        # Draw by iteraring through each outer cell, column-wise then row-wise
        coordinate_map = CONSTANT.COORDINATE_MAP
        first_sector_y, last_sector_y = coordinate_map.Y_RANGE
        first_sector_x, last_sector_x = coordinate_map.X_RANGE
        sector_size_px = CONSTANT.SCREEN.SECTOR_SIZE_PX
        background_color = "white"
        for sector_x in range(first_sector_x, last_sector_x + 1):
            draw_rectangle_on_surface(
                surface=surface,
                x=sector_x * sector_size_px,
                y=first_sector_y,
                width=sector_size_px,
                height=sector_size_px,
                background_color=background_color,
                text=str(sector_x),
            )

        for sector_y in range(first_sector_y, last_sector_y + 1):
            draw_rectangle_on_surface(
                surface=surface,
                x=first_sector_x,
                y=sector_y * sector_size_px,
                width=sector_size_px,
                height=sector_size_px,
                background_color=background_color,
                text=str(sector_y),
            )

    def __draw_lanes_sectors(self, surface: pygame.Surface):

        # Determine drawing plane for lanes
        first_sector_y = CONSTANT.TERMINAL_FLOOR.START_SECTOR_Y
        last_sector_y = first_sector_y + CONSTANT.TERMINAL_FLOOR.NUMBER_OF_SECTORS_Y

        first_sector_x = CONSTANT.TERMINAL_FLOOR.START_SECTOR_X
        last_sector_x = first_sector_x + CONSTANT.TERMINAL_FLOOR.NUMBER_OF_SECTORS_X

        # Create mapping of background color based on sector_y
        bg_color_map = {
            **{
                y: CONSTANT.TERMINAL_FLOOR.QC_TRAVEL_LANE.COLOR
                for y in CONSTANT.TERMINAL_FLOOR.QC_TRAVEL_LANE.SECTOR_RANGE_Y
            },
            **{
                y: CONSTANT.TERMINAL_FLOOR.BUFFER_ZONE_LANE.COLOR
                for y in CONSTANT.TERMINAL_FLOOR.BUFFER_ZONE_LANE.SECTOR_RANGE_Y
            },
            **{
                y: CONSTANT.TERMINAL_FLOOR.HIGHWAY_LANE.COLOR
                for y in CONSTANT.TERMINAL_FLOOR.HIGHWAY_LANE.SECTOR_RANGE_Y
            },
        }

        # Draw by iteraring through each cell, column-wise per row
        sector_size_px = CONSTANT.SCREEN.SECTOR_SIZE_PX
        for sector_y in range(first_sector_y, last_sector_y):
            for sector_x in range(first_sector_x, last_sector_x):
                background_color = bg_color_map.get(sector_y, "white")

                draw_rectangle_on_surface(
                    surface=surface,
                    x=sector_x * sector_size_px,
                    y=sector_y * sector_size_px,
                    width=sector_size_px,
                    height=sector_size_px,
                    background_color=background_color,
                )

        # Draw arrows indicating permitted directions
        buffer = CONSTANT.ARROW.MARGIN
        arrow_length = CONSTANT.ARROW.LENGTH
        sector_size_px = CONSTANT.SCREEN.SECTOR_SIZE_PX

        right_arrows = [
            (sector_size_px + buffer, cell_x * sector_size_px + buffer)
            for cell_x in CONSTANT.TERMINAL_FLOOR.RIGHT_ARROW_SECTOR_X
        ]
        for arrow in right_arrows:
            arrow_start_x, arrow_start_y = arrow
            draw_arrow_on_surface(
                surface,
                arrow_start_x,
                arrow_start_y,
                length=arrow_length,
                color=CONSTANT.TERMINAL_FLOOR.RIGHT_ARROW_COLOR,
                direction="right",
            )

        left_arrows = [
            (sector_size_px + sector_size_px - buffer, cell_x * sector_size_px + buffer)
            for cell_x in CONSTANT.TERMINAL_FLOOR.LEFT_ARROW_SECTOR_X
        ]
        for arrow in left_arrows:
            arrow_start_x, arrow_start_y = arrow
            draw_arrow_on_surface(
                surface,
                arrow_start_x,
                arrow_start_y,
                length=arrow_length,
                color=CONSTANT.TERMINAL_FLOOR.LEFT_ARROW_COLOR,
                direction="left",
            )

    def __draw_quay_crane_sectors(self, surface: pygame.Surface):
        quay_crane_floor = CONSTANT.QUAY_CRANE_FLOOR
        sector_size_px = CONSTANT.SCREEN.SECTOR_SIZE_PX
        sector_x_list = list(
            range(
                quay_crane_floor.START_SECTOR_X,
                CONSTANT.TERMINAL_FLOOR.NUMBER_OF_SECTORS_X,
                quay_crane_floor.NUMBER_OF_SECTORS_BETWEEN_QC,
            )
        )
        start_sector_y = quay_crane_floor.START_SECTOR_Y

        for sector_x, qc_name in zip(sector_x_list, quay_crane_floor.QC_NAMES):
            # draw QC
            draw_image_on_surface(
                surface=surface,
                filepath=quay_crane_floor.IMAGE,
                x=sector_x * sector_size_px,
                y=sector_size_px,
            )

            # draw QC name
            draw_rectangle_on_surface(
                surface=surface,
                x=sector_x * sector_size_px,
                y=start_sector_y * sector_size_px,
                width=2 * sector_size_px,
                height=sector_size_px,
                background_color=quay_crane_floor.QC_NAME_SECTOR_COLOR,
                text=qc_name,
            )

            # draw IN
            draw_rectangle_on_surface(
                surface=surface,
                x=sector_x * sector_size_px,
                y=(start_sector_y + 1) * sector_size_px,
                width=sector_size_px,
                height=sector_size_px,
                background_color=quay_crane_floor.IN_SECTOR_COLOR,
                text="IN",
            )

            # draw OUT
            draw_rectangle_on_surface(
                surface=surface,
                x=(sector_x + 1) * sector_size_px,
                y=(start_sector_y + 1) * sector_size_px,
                width=sector_size_px,
                height=sector_size_px,
                background_color=quay_crane_floor.OUT_SECTOR_COLOR,
                text="OUT",
            )

            # draw arrow
            buffer = CONSTANT.ARROW.MARGIN
            arrow_length = CONSTANT.ARROW.LENGTH
            draw_arrow_on_surface(
                surface=surface,
                x=sector_x * sector_size_px + buffer,
                y=int((4 + 1 / 2) * sector_size_px - buffer),
                length=arrow_length,
                color=quay_crane_floor.ARROW_COLOR,
                direction="up",
            )

            draw_arrow_on_surface(
                surface=surface,
                x=int((sector_x + 1 / 2) * sector_size_px + buffer),
                y=3 * sector_size_px + buffer,
                length=arrow_length,
                color=quay_crane_floor.ARROW_COLOR,
                direction="right",
            )

            draw_arrow_on_surface(
                surface=surface,
                x=int((sector_x + 3 / 2) * sector_size_px + buffer),
                y=int((3 + 1 / 2) * sector_size_px + buffer),
                length=arrow_length,
                color=quay_crane_floor.ARROW_COLOR,
                direction="down",
            )

    def __draw_yard_sectors(self, surface: pygame.Surface):

        yard_floor = CONSTANT.YARD_FLOOR
        sector_size_px = CONSTANT.SCREEN.SECTOR_SIZE_PX
        start_yard_y = yard_floor.START_SECTOR_Y
        sector_x_list = list(
            range(
                yard_floor.START_SECTOR_X,
                CONSTANT.TERMINAL_FLOOR.NUMBER_OF_SECTORS_X - 1,
                yard_floor.NUMBER_OF_SECTORS_BETWEEN_YARD_BLOCK,
            )
        )

        for idx, sector_x in enumerate(sector_x_list):
            yard_names = [
                yard_floor.YARD_NAMES[idx * 2],
                yard_floor.YARD_NAMES[idx * 2 + 1],
            ]
            for x_offset, yard_name in zip([0, 2], yard_names):
                # draw IN
                draw_rectangle_on_surface(
                    surface=surface,
                    x=(sector_x + x_offset) * sector_size_px,
                    y=start_yard_y * sector_size_px,
                    width=sector_size_px,
                    height=sector_size_px,
                    background_color=yard_floor.IN_SECTOR_COLOR,
                    text="IN",
                )

                # draw OUT
                draw_rectangle_on_surface(
                    surface=surface,
                    x=(sector_x + 1 + x_offset) * sector_size_px,
                    y=start_yard_y * sector_size_px,
                    width=sector_size_px,
                    height=sector_size_px,
                    background_color=yard_floor.OUT_SECTOR_COLOR,
                    text="OUT",
                )

                # draw Yard name
                draw_rectangle_on_surface(
                    surface=surface,
                    x=(sector_x + x_offset) * sector_size_px,
                    y=(start_yard_y + 1) * sector_size_px,
                    width=2 * sector_size_px,
                    height=sector_size_px,
                    background_color="white",
                    text=yard_name,
                )

                # draw Yard
                draw_image_on_surface(
                    surface=surface,
                    filepath=yard_floor.IMAGE,
                    x=(sector_x + x_offset) * sector_size_px,
                    y=(start_yard_y + 2) * sector_size_px,
                )

                # draw Arrow
                buffer = CONSTANT.ARROW.MARGIN
                arrow_length = CONSTANT.ARROW.LENGTH

                draw_arrow_on_surface(
                    surface=surface,
                    x=(sector_x + x_offset) * sector_size_px + buffer,
                    y=int((start_yard_y - 1 / 2) * sector_size_px + buffer),
                    length=arrow_length,
                    color=yard_floor.ARROW_COLOR,
                    direction="down",
                )
                draw_arrow_on_surface(
                    surface=surface,
                    x=(sector_x + 1 + x_offset) * sector_size_px - 2 * buffer,
                    y=(start_yard_y + 1) * sector_size_px - buffer,
                    length=arrow_length,
                    color=yard_floor.ARROW_COLOR,
                    direction="right",
                )
                draw_arrow_on_surface(
                    surface=surface,
                    x=(sector_x + 2 + x_offset) * sector_size_px - buffer,
                    y=(start_yard_y + 1 / 2) * sector_size_px - buffer,
                    length=arrow_length,
                    color=yard_floor.ARROW_COLOR,
                    direction="up",
                )

    def __draw_job_tracker(self, screen: Any):
        ui_job_tracker = CONSTANT.JOB_TRACKER

        # Draw outer table
        sector_size_px = CONSTANT.SCREEN.SECTOR_SIZE_PX
        draw_rectangle_on_screen(
            screen=screen,
            x=ui_job_tracker.START_SECTOR_X * sector_size_px,
            y=ui_job_tracker.START_SECTOR_Y * sector_size_px,
            width=ui_job_tracker.WIDTH_PX,
            height=ui_job_tracker.HEIGHT_PX,
        )

        # Draw individual cells: JOBS, QUAY_CRANE, HT, YARD
        cell_infos = [
            ui_job_tracker.JOB_INFO,
            # ui_job_tracker.QUAY_CRANE_INFO,
            ui_job_tracker.HT_INFO,
            # ui_job_tracker.YARD_INFO,
            ui_job_tracker.HT_LOCATIONS,
        ]
        cell_statuses = [
            self.job_tracker.get_job_status(),
            # self.job_tracker.get_quay_crane_status(),
            self.job_tracker.get_HT_status(),
            # self.job_tracker.get_yard_status(),
            self.job_tracker.get_HT_locations(),
        ]

        for cell, status in zip(cell_infos, cell_statuses):
            draw_rectangle_on_screen(
                screen=screen,
                x=cell.CELL_X_RANGE[0],
                y=cell.CELL_Y_RANGE[0],
                width=cell.CELL_X_RANGE[1] - cell.CELL_X_RANGE[0],
                height=cell.CELL_Y_RANGE[1] - cell.CELL_Y_RANGE[0],
            )
            draw_text_on_screen(
                screen,
                text=f" \n{cell.HEADER}\n ",
                x_range=cell.CELL_X_RANGE,
                y_range=cell.HEADER_Y_RANGE,
                align=("center", "middle"),
                color="cyan",
                font=self.tracker_font,
            )
            if cell.HEADER == "HT LOCATIONS":
                draw_text_on_screen(
                    screen,
                    text=status,
                    x_range=cell.CELL_X_RANGE,
                    y_range=cell.ATTR_Y_RANGE,
                    align=("center", "middle"),
                    color="cyan",
                    line_spacing=5,
                    font=self.tracker_font,
                )
            else:
                draw_text_on_screen(
                    screen,
                    text=status,
                    x_range=cell.CELL_X_RANGE,
                    y_range=cell.ATTR_Y_RANGE,
                    align=("left", "middle"),
                    color="cyan",
                    line_spacing=5,
                    font=self.tracker_font,
                )

    def update(self, terminal_statistics: Dict[str, Any]):
        self.job_tracker.update_data(terminal_statistics)

    def draw(self, screen: Any):
        # render simulation visuals
        screen.surface.blit(self.surface, (0, 0))

        self.__draw_job_tracker(screen)

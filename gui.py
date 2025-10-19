import os
import sys

import pgzrun
import pygame
from logzero import logger

from src.constant import CONSTANT
from src.simulation import Simulation
from src.ui.HT import HTActorFleet
from src.ui.terminal import TerminalDisplay
from src.utils import logging

# intializes font for text rendering
pygame.font.init()

# Force window to be centered
os.environ["SDL_VIDEO_CENTERED"] = "1"
WIDTH = CONSTANT.SCREEN.WIDTH_PX
HEIGHT = CONSTANT.SCREEN.HEIGHT_PX

# Configurable parameters
number_of_refresh_per_physical_second = 5
refresh_rate = 1 / number_of_refresh_per_physical_second

# Flags
is_planning_in_progress = False  # track if the planning compute is still running

# Initialize program
logging.setup_logger()
terminal_display = TerminalDisplay()
HT_fleet = HTActorFleet()
simulation = Simulation()


def draw():
    screen.clear()
    terminal_display.draw(screen)
    HT_fleet.draw()


def process():
    global is_planning_in_progress

    # Skip if already running
    if is_planning_in_progress:
        return

    # Ensure at most 1 process() thread can run at any time
    is_planning_in_progress = True

    # Execute one simulation move
    simulation.update()
    system_time = simulation.get_current_time()

    # Display HT actors
    HT_name_coords_map = simulation.export_HT_name_coords()
    HT_fleet.update(HT_name_coords_map)

    # Only update tracking when the move involves planning
    if (system_time % CONSTANT.PLANNING_INTERVAL) == 0:
        terminal_statistics = simulation.export_terminal_statistics()
        terminal_display.update(terminal_statistics)

    # Terminate the simulation if all jobs are completed
    if simulation.has_completed_all_jobs():
        logger.info(f"Time: {system_time:,.0f}. All jobs are completed. Stop.")
        simulation.export_job_report()
        sys.exit()
    # or deadlock detected
    if simulation.has_deadlock():
        logger.info(
            f"Time: {system_time:,.0f}. Dead-lock detected. No HT moves in the past {CONSTANT.DEADLOCK_THRESHOLD} secs."
        )
        simulation.export_job_report()
        sys.exit()

    # Release process() and signal new thread can be created
    is_planning_in_progress = False


def update():
    # Always run (every frame)
    # nil

    # Schedule logic (run once to set up interval-based updates)
    if not hasattr(update, "scheduled"):
        clock.schedule_interval(process, refresh_rate)
        update.scheduled = True


pgzrun.go()

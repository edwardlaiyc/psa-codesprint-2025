from logzero import logger

from src.constant import CONSTANT
from src.simulation import Simulation
from src.utils import logging

if __name__ == "__main__":
    logging.setup_logger()
    simulation = Simulation()
    while True:
        simulation.update()
        system_time = simulation.get_current_time()
        if simulation.has_completed_all_jobs():
            logger.info(f"Time: {system_time:,.0f}. All jobs are completed. Stop.")
            break
        if simulation.has_deadlock():
            logger.info(
                f"Time: {system_time:,.0f}. Dead-lock detected. No HT moves in the past {CONSTANT.DEADLOCK_THRESHOLD} secs."
            )
            break
    simulation.export_job_report()

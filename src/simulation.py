from collections import namedtuple
from typing import Any, Dict

import pandas as pd
from logzero import logger

from src.constant import CONSTANT
from src.floor import Coordinate, SectorMap, SectorMapSnapshot
from src.operate.engine import OperationEngine
from src.operators import (
    HT_Coordinate_View,
    HTOperator,
    QCOperator,
    YardOperator,
)
from src.plan.engine import PlanningEngine


class Simulation:
    """
    A class to initialize and coordinate the components required for simulating the port terminal process.

    This class sets up the planning and operation engines using resources created for operations
    and monitoring. It also reads job data from an input CSV file to feed into the planning engine.

    Attributes
    ----------
    planning_engine : PlanningEngine
        An instance of the planning engine initialized with job data and monitoring resources.
    operation_engine : OperationEngine
        An instance of the operation engine initialized with operation and monitoring resources.
    operation_resources : dict
        Resources created for operational use.
    monitoring_resources : dict
        Resources created for monitoring operational performance.
    plan_countdown : int
        A countdown timer used for scheduling planning operations.
    """

    def __init__(self):
        operation_resources = self.create_operation_resources()
        monitoring_resources = self.create_monitoring_resources(operation_resources)

        input_df = pd.read_csv("data/input.csv", header=0)
        self.planning_engine = PlanningEngine(
            job_df=input_df, monitoring_resources=monitoring_resources
        )
        self.operation_engine = OperationEngine(
            operation_resources=operation_resources,
            monitoring_resources=monitoring_resources,
        )
        self.operation_resources = operation_resources
        self.monitoring_resources = monitoring_resources
        self.plan_countdown: int = 0

    def update(self):
        """Primary class to trigger the simulation process per one time unit"""
        if self.is_planning_due():
            # PLANNING
            # logger.info("Planning -> Operating")
            self.planning_engine.fetch_job_status()
            new_jobs = self.planning_engine.plan()

            # OPERATING
            # logger.info("Entered operating")
            self.operation_engine.add_new_jobs(new_jobs)
            self.operation_engine.operate()
        else:
            # logger.info("Operating(only)")
            self.operation_engine.operate()

    def is_planning_due(self) -> bool:
        if self.plan_countdown <= 0:
            self.plan_countdown = CONSTANT.PLANNING_INTERVAL
            return True

        self.plan_countdown -= CONSTANT.JOB_PARAMETER.SYSTEM_TIME_PASSED
        return False

    def create_operation_resources(self) -> namedtuple:
        # create sectors
        sector_map = SectorMap()

        # create HT resources
        HT_resource_group = {
            HT_name: HTOperator(
                name=HT_name, coord=Coordinate(location[0], location[1])
            )
            for HT_name, location in zip(
                CONSTANT.HT_FLEET.HT_NAMES, CONSTANT.HT_FLEET.HT_INIT_COORDINATES
            )
        }
        # put HT locations to track on SectorMap
        for HT_name, HT_operator in HT_resource_group.items():
            sector_map.add_occupator(
                coord=HT_operator.get_coordinate(), HT_name=HT_name
            )

        # create QC resources
        QC_resource_group = {
            QC_name: QCOperator(name=QC_name)
            for QC_name in CONSTANT.QUAY_CRANE_FLOOR.QC_NAMES
        }

        # create Yard resources
        yard_resource_group = {
            yard_name: YardOperator(name=yard_name)
            for yard_name in CONSTANT.YARD_FLOOR.YARD_NAMES
        }

        OperationResources = namedtuple(
            "OperationResources",
            [
                "sector_map",
                "HT_resource_group",
                "QC_resource_group",
                "yard_resource_group",
            ],
        )
        return OperationResources(
            sector_map, HT_resource_group, QC_resource_group, yard_resource_group
        )

    def create_monitoring_resources(
        self, operation_resources: namedtuple
    ) -> namedtuple:
        sector_map_snapshot = SectorMapSnapshot(operation_resources.sector_map)
        HT_coord_tracker = HT_Coordinate_View(operation_resources.HT_resource_group)

        MonitoringResources = namedtuple(
            "MonitoringResources", ["sector_map_snapshot", "HT_coord_tracker"]
        )
        return MonitoringResources(sector_map_snapshot, HT_coord_tracker)

    def has_completed_all_jobs(self) -> bool:
        if self.planning_engine.is_all_job_completed():

            return True
        return False

    def has_deadlock(self) -> bool:
        if self.planning_engine.is_deadlock():
            return True
        return False

    def export_terminal_statistics(self) -> Dict[str, Dict[str, int]]:
        """Extract terminal status to feed into JobProgress in UI."""
        # jobs
        number_of_total_jobs = 20000
        number_of_completed_jobs = self.planning_engine.get_number_of_completed_jobs()
        number_of_remaining_jobs = number_of_total_jobs - number_of_completed_jobs
        current_time = self.operation_engine.get_current_time()

        # QC
        number_of_total_QC = 8
        number_of_idle_QC = sum(
            [
                not QC_ops.is_displayed_busy()
                for _, QC_ops in self.operation_resources.QC_resource_group.items()
            ]
        )
        number_of_active_QC = number_of_total_QC - number_of_idle_QC

        # HT
        number_of_total_HT = 80
        number_of_non_moving_HT = self.planning_engine.get_non_moving_HT()
        number_of_moving_HT = number_of_total_HT - number_of_non_moving_HT

        # Yard
        number_of_total_yard = 16
        number_of_idle_yard = sum(
            [
                not yard_ops.is_displayed_busy()
                for _, yard_ops in self.operation_resources.yard_resource_group.items()
            ]
        )
        number_of_active_yard = number_of_total_yard - number_of_idle_yard

        # HT Locations
        HT_at_QC_locations = [
            [HT_name, str(HT_ops.get_coordinate())]
            for HT_name, HT_ops in self.operation_resources.HT_resource_group.items()
            if HT_ops.get_coordinate().y == 3
        ]
        HT_at_yard_locations = [
            [HT_name, str(HT_ops.get_coordinate())]
            for HT_name, HT_ops in self.operation_resources.HT_resource_group.items()
            if HT_ops.get_coordinate().y == 13
        ]
        HT_at_other_locations = [
            [HT_name, str(HT_ops.get_coordinate())]
            for HT_name, HT_ops in self.operation_resources.HT_resource_group.items()
            if HT_ops.get_coordinate().y not in [3, 13]
        ]
        HT_locations = HT_at_QC_locations + HT_at_yard_locations + HT_at_other_locations

        data = {
            "JOBS": {
                "TOTAL": number_of_total_jobs,
                "REMAINING": number_of_remaining_jobs,
                "COMPLETED": number_of_completed_jobs,
                "TIME(secs)": current_time,
            },
            "QUAY CRANE": {
                "TOTAL": number_of_total_QC,
                "ACTIVE": number_of_active_QC,
                "IDLE": number_of_idle_QC,
            },
            "HT": {
                "TOTAL": number_of_total_HT,
                "MOVING": number_of_moving_HT,
                "NOT MOVING": number_of_non_moving_HT,
            },
            "YARD": {
                "TOTAL": number_of_total_yard,
                "ACTIVE": number_of_active_yard,
                "IDLE": number_of_idle_yard,
            },
            "HT LOCATIONS": HT_locations,
        }
        return data

    def export_HT_name_coords(self) -> Dict[str, Coordinate]:
        """Extract latest locations of HTs to display on UI."""
        HT_name_coords_map = dict()
        sector_size_px = CONSTANT.SCREEN.SECTOR_SIZE_PX

        for HT_name, HT_operator in self.operation_resources.HT_resource_group.items():
            coord = HT_operator.get_coordinate()
            # convert from terminal to simulation coordinate
            x_px = coord.x * sector_size_px + sector_size_px / 2
            y_px = coord.y * sector_size_px + sector_size_px / 2

            HT_name_coords_map[HT_name] = Coordinate(x_px, y_px)

        return HT_name_coords_map

    def export_job_report(self):
        """Export all jobs attributes with start/end time as report."""
        output_df = self.planning_engine.export_job_report()
        filepath = "data/output.csv"
        output_df.to_csv(filepath, index=False)
        logger.info(f"Output job report: {filepath}")

    def get_current_time(self):
        return self.operation_engine.get_current_time()

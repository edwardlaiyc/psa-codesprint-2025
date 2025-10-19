from collections import namedtuple
from typing import Dict, List

from logzero import logger

from src.constant import CONSTANT
from src.floor import SectorMap
from src.job import InstructionType, Job, Status
from src.operators import (
    HT_Coordinate_View,
    HTOperator,
    QCOperator,
    ResourceOperator,
    YardOperator,
)


class JobQueue:
    def __init__(self):
        self.job_items: List[Job] = list()

    def push(self, job: Job):
        self.job_items.append(job)

    def pop_by_job_seq(self, job_seq: str):
        for i, job in enumerate(self.job_items):
            if job_seq == job.get_job_info().get("QC_job_sequence"):
                self.job_items.pop(i)
                return
        raise ValueError(f"Job Sequence {job_seq} not found.")

    def get_job_by_job_seq(self, job_seq: str):
        for i, job in enumerate(self.job_items):
            if job_seq == job.get_job_info().get("QC_job_sequence"):
                return job
        raise ValueError(f"Job Sequence {job_seq} not found.")

    def is_empty(self):
        return len(self.job_items) == 0

    def size(self):
        return len(self.job_items)

    def __iter__(self):
        return iter(self.job_items)


class OperationEngine:
    """
    Manages operational resources and coordinates handling teams, yard operators, and QC operators
    within the sector map environment.

    Parameters
    ----------
    operation_resources : namedtuple
        A namedtuple containing operational resources, including HT resource group,
        yard resource group, QC resource group, and the sector map.
    monitoring_resources : namedtuple
        A namedtuple containing monitoring resources such as HT coordinate tracker.

    Attributes
    ----------
    HT_resource_group : Dict[str, HTOperator]
        Mapping of HT names to their operator instances.
    yard_resource_group : Dict[str, YardOperator]
        Mapping of yard operator names to their instances.
    QC_resource_group : Dict[str, QCOperator]
        Mapping of QC operator names to their instances.
    sector_map : SectorMap
        The sector map representing the operational environment.
    HT_coord_tracker : HT_Coordinate_View
        Tracker for HT coordinates.
    job_queue : JobQueue
        Queue managing jobs to be processed.
    time_counter : int
        Counter tracking the elapsed operational time or ticks.
    """

    def __init__(
        self,
        operation_resources: namedtuple,
        monitoring_resources: namedtuple,
    ):

        self.HT_resource_group: Dict[str, HTOperator] = (
            operation_resources.HT_resource_group
        )
        self.yard_resource_group: Dict[str, YardOperator] = (
            operation_resources.yard_resource_group
        )
        self.QC_resource_group: Dict[str, QCOperator] = (
            operation_resources.QC_resource_group
        )
        self.sector_map: SectorMap = operation_resources.sector_map
        self.HT_coord_tracker: HT_Coordinate_View = (
            monitoring_resources.HT_coord_tracker
        )
        self.job_queue: JobQueue = JobQueue()
        self.time_counter: int = 0

    def add_new_jobs(self, new_jobs: List[Job]):
        for job in new_jobs:
            job_info = job.get_job_info()
            # logger.debug(f"New job pushed in queue: {job}")
            self.job_queue.push(job)

    def operate(self):
        # logger.info("Assign new job: First come first serve")

        # Ensure all jobs sequences are in order by QC then by job_sequence
        for i, j in zip(self.job_queue.job_items[:-1], self.job_queue.job_items[1:]):
            QC_i, number_i = i.get_job_info()["QC_job_sequence"].split("_")
            QC_j, number_j = j.get_job_info()["QC_job_sequence"].split("_")
            if (QC_i == QC_j) and (number_i > number_j):
                raise ValueError(f"This should not happen.")

        for job in self.job_queue:
            job_info = job.get_job_info()

            # Skip if job has started
            job_status = job_info["job_status"]
            if job_status != Status.NOT_STARTED:
                continue

            # Skip if HT operator is not available
            HT_name = job_info["assigned_HT_name"]
            HT_operator = self.HT_resource_group.get(HT_name)
            if not HT_operator.is_available():
                continue

            # Chope HT resource and kickstart the job
            job_seq = job_info["QC_job_sequence"]
            HT_operator.lock(job_seq)
            job.start_job(self.time_counter)
            job.chope_HT()

        # logger.info("Pick up tasks in IN-PROGRESS jobs to execute.")
        # Emulate the passing of SYSTEM TIME
        self.time_counter += CONSTANT.JOB_PARAMETER.SYSTEM_TIME_PASSED
        # logger.info(f"Time counter: {self.time_counter}")
        HT_drive_operator_map: Dict[str, HTOperator] = (
            dict()
        )  # collect DRIVE tasks to execute separately in specific order
        for job in self.job_queue:
            job_info = job.get_job_info()
            job_status = job_info["job_status"]
            if job_status != Status.IN_PROGRESS:
                continue
            instruction = job.get_latest_instruction()
            instruction_type = instruction.get_instruction_type()
            # logger.debug(
            #     f"Assigned: job_seq={job_seq}, job_type={job_type}, instruction={instruction}"
            # )

            # Retrieve corresponding resources
            QC_name = job_info["QC_name"]
            QC_operator = self.QC_resource_group.get(QC_name)
            HT_name = job_info["assigned_HT_name"]
            HT_operator = self.HT_resource_group.get(HT_name)
            yard_name = job_info["assigned_yard_name"]
            yard_operator = self.yard_resource_group.get(yard_name)

            job_seq = job_info["QC_job_sequence"]

            # BOOKING TASKS:
            # Book, check its turn and proceed to next instruction if ready
            if instruction_type == InstructionType.BOOK_QC:
                # book QC service: it is nullified if does not follow job sequence order
                if not QC_operator.is_in_queue(job_seq):
                    QC_operator.join_queue(job_seq)

                # when it's the second in queue, can proceed to next instruction
                if QC_operator.is_near_turn(job_seq):
                    job.proceed_to_next_instruction(timestamp=self.time_counter)
                    continue

            if instruction_type == InstructionType.BOOK_YARD:
                # book Yard service
                if not yard_operator.is_in_queue(job_seq):
                    yard_operator.join_queue(job_seq)

                # when it's the third in queue, can proceed to next instruction
                if yard_operator.is_near_turn(job_seq):
                    job.proceed_to_next_instruction(timestamp=self.time_counter)
                    continue

            # NON-BOOKING TASKS:
            # Chope the resource once available (for new task) and start or resume tasks
            if not instruction.has_started():
                if instruction_type == InstructionType.DRIVE:
                    HT_operator.receive_task(planned_path=instruction.get_paths())
                    instruction.set_start_time(timestamp=self.time_counter)

                if (instruction_type == InstructionType.WORK_QC) and (
                    QC_operator.is_available()
                ):
                    if QC_operator.is_ready_to_serve(job_seq):
                        QC_operator.lock(job_seq)
                        QC_operator.receive_task()
                        job.chope_QC()
                        instruction.set_start_time(timestamp=self.time_counter)

                if (instruction_type == InstructionType.WORK_YARD) and (
                    yard_operator.is_available()
                ):
                    if yard_operator.is_ready_to_serve(job_seq):
                        yard_operator.lock(job_seq)
                        yard_operator.receive_task()
                        job.chope_yard()
                        instruction.set_start_time(timestamp=self.time_counter)

            # Execute task
            # Exception: DRIVE operation to happen in code block right below
            if instruction_type == InstructionType.DRIVE:
                HT_drive_operator_map[HT_name] = HT_operator

            # Execute if this is the job QC is locked by
            if instruction_type == InstructionType.WORK_QC:
                if QC_operator.get_job_seq() == job_seq:
                    QC_operator.execute_task()
                    self.mark_instruction_progress_and_release_operator_if_applicable(
                        QC_operator
                    )

            # Execute if this is the job yard is locked by
            if instruction_type == InstructionType.WORK_YARD:
                if yard_operator.get_job_seq() == job_seq:
                    yard_operator.execute_task()
                    self.mark_instruction_progress_and_release_operator_if_applicable(
                        yard_operator
                    )

        HT_sorted_names = sorted(HT_drive_operator_map.keys())
        # logger.info(
        #     f"Execute Drive task first, with priority for smaller serial no HT: {len(HT_sorted_names)} tasks."
        # )
        for HT_name in HT_sorted_names:
            # check planned coordinate
            HT_operator = HT_drive_operator_map.get(HT_name)
            current_coord = HT_operator.get_coordinate()
            planned_coord = HT_operator.get_planned_coordinate()

            # proceed if sector has availability
            if self.sector_map.is_sector_available(planned_coord):
                HT_operator.execute_task()
                # update map tracking accordingly
                self.sector_map.move_occupator(
                    from_coord=current_coord, to_coord=planned_coord, HT_name=HT_name
                )
                self.mark_instruction_progress_and_release_operator_if_applicable(
                    HT_operator
                )

        ### CLEAN UP QUEUE
        for job in self.job_queue:
            job_info = job.get_job_info()
            job_seq = job_info["QC_job_sequence"]
            if job.is_completed():
                self.job_queue.pop_by_job_seq(job_seq)

    def mark_instruction_progress_and_release_operator_if_applicable(
        self,
        operator: ResourceOperator,
    ):
        if operator.has_completed_task():
            job_seq = operator.get_job_seq()
            job = self.job_queue.get_job_by_job_seq(job_seq)

            job.proceed_to_next_instruction(timestamp=self.time_counter)
            if (type(operator) is HTOperator) and (not job.is_HT_required()):
                operator.release(job_seq)
            if (type(operator) is QCOperator) and (not job.is_QC_required()):
                operator.release(job_seq)
            if (type(operator) is YardOperator) and (not job.is_yard_required()):
                operator.release(job_seq)

    def get_number_of_in_progress_jobs(self):
        return self.job_queue.size()

    def get_current_time(self):
        return self.time_counter

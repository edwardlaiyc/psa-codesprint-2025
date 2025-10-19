from typing import Dict, List, Optional

from src.constant import CONSTANT
from src.floor import Coordinate


class ResourceOperator:
    def __init__(self, name: str):
        self.name: str = name
        self.locked_by: Optional[str] = None  # a job sequence. Ex: QC1_0001

    def is_available(self):
        return self.locked_by is None

    def lock(self, job_seq: str):
        if self.locked_by is not None:
            raise ValueError("Resource is being used. Cannot lock.")
        self.locked_by = job_seq

    def release(self, job_seq: str):
        if self.locked_by != job_seq:
            raise ValueError(f"Job Seq {job_seq} is not active resource holder.")
        self.locked_by = None

    def get_job_seq(self):
        return self.locked_by

    def receive_task(self):
        raise NotImplementedError("Not implemented.")

    def is_working_on_task(self):
        raise NotImplementedError("Not implemented.")

    def execute_task(self):
        raise NotImplementedError("Not implemented.")

    def has_completed_task(self):
        raise NotImplementedError("Not implemented.")


class QueuedResourceOperator(ResourceOperator):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.queue: List[str] = list()
        self.near_turn_limit: int = 1

    def join_queue(self, job_seq: str):
        if job_seq not in self.queue:
            self.queue.append(job_seq)

    def is_in_queue(self, job_seq: str):
        return job_seq in self.queue

    def lock(self, job_seq: str):
        if self.locked_by is not None:
            raise ValueError("Resource is being used. Cannot lock.")

        if (len(self.queue) == 0) or (self.queue[0] != job_seq):
            raise LookupError(
                "Queue is empty or job sequence is not the first in queue."
            )
        self.locked_by = job_seq

    def release(self, job_seq: str):
        if self.locked_by != job_seq:
            raise ValueError(f"Job Seq {job_seq} is not active resource holder.")
        self.locked_by = None
        self.queue.pop(0)

    def is_near_turn(self, job_seq: str) -> bool:
        if job_seq in self.queue[: self.near_turn_limit]:
            return True
        return False

    def is_ready_to_serve(self, job_seq: str):
        if len(self.queue) == 0:
            raise ValueError("No one in queue.")
        return job_seq == self.queue[0]


class YardOperator(QueuedResourceOperator):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.near_turn_limit = 3
        self.__handling_task_progress: int = None

    def is_displayed_busy(self):
        is_choped = not self.is_available()
        return is_choped

    def release(self, job_seq: str):
        super().release(job_seq)
        self.__handling_task_progress = None

    def receive_task(self):
        self.__handling_task_progress = 0

    def is_working_on_task(self) -> bool:
        return True if self.__handling_task_progress is not None else False

    def execute_task(self):
        # do nothing if receive no task
        if not self.is_working_on_task():
            return

        # execute the work
        if not self.has_completed_task():
            self.__handling_task_progress += CONSTANT.JOB_PARAMETER.SYSTEM_TIME_PASSED

    def has_completed_task(self):
        if (
            self.__handling_task_progress
            >= CONSTANT.JOB_PARAMETER.YARD_WORK_TIME_REQUIRED
        ):
            return True
        return False

    def __str__(self):
        return f"YardOperator(name={self.name}, locked_by={self.locked_by}, progress={self.__handling_task_progress})"


class QCOperator(QueuedResourceOperator):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.near_turn_limit = 2
        self.__handling_task_progress: int = None
        self.expected_minimum_seq_number: int = 0

    def is_displayed_busy(self):
        is_choped = not self.is_available()
        return is_choped

    def join_queue(self, job_seq: str):
        new_seq_number = int(job_seq.split("_")[1])

        # if queue is empty, only add if sequence number is right after most recent completed one
        if len(self.queue) == 0:
            if self.expected_minimum_seq_number + 1 == new_seq_number:
                self.queue.append(job_seq)

        else:
            # only add to queue if not already in queue
            if job_seq not in self.queue:
                latest_job_seq_in_queue = self.queue[-1]
                QC_name = latest_job_seq_in_queue.split("_")[0]
                latest_seq_number = int(latest_job_seq_in_queue.split("_")[1])
                expected_job_seq = f"{QC_name}_{latest_seq_number+1:04d}"

                # only add to queue when follow proper sequence
                if job_seq == expected_job_seq:
                    self.queue.append(job_seq)

    def release(self, job_seq: str):
        super().release(job_seq)
        self.__handling_task_progress = None
        self.expected_minimum_seq_number = int(job_seq.split("_")[1])

    def receive_task(self):
        self.__handling_task_progress = 0

    def is_working_on_task(self) -> bool:
        return True if self.__handling_task_progress is not None else False

    def execute_task(self):
        # do nothing if receive no task
        if not self.is_working_on_task():
            return

        # execute the work
        if not self.has_completed_task():
            self.__handling_task_progress += CONSTANT.JOB_PARAMETER.SYSTEM_TIME_PASSED

    def has_completed_task(self):
        if (
            self.__handling_task_progress
            >= CONSTANT.JOB_PARAMETER.QC_WORK_TIME_REQUIRED
        ):
            return True
        return False

    def __str__(self):
        return f"QCOperator(name={self.name}, locked_by={self.locked_by}, progress={self.__handling_task_progress}, expected_min_seq={self.expected_minimum_seq_number})"


class HTOperator(ResourceOperator):
    def __init__(self, name: str, coord: Coordinate):
        super().__init__(name=name)
        self.coord: Coordinate = coord
        self.planned_path: List[Coordinate] = list()
        self.path_step: int = None

    def is_displayed_busy(self):
        is_choped = not self.is_available()
        return is_choped

    def release(self, job_seq: str):
        super().release(job_seq)
        self.planned_path = list()
        self.path_step = None

    def receive_task(self, planned_path: List[Coordinate]):
        self.planned_path = planned_path
        self.path_step = 0

    def is_working_on_task(self) -> bool:
        return True if self.path_step is not None else False

    def execute_task(self):
        # do nothing if receive no task
        if not self.is_working_on_task():
            return

        # execute the work
        if not self.has_completed_task():
            next_coord = self.planned_path[self.path_step]
            self.coord = next_coord
            self.path_step += 1

    def has_completed_task(self):
        if self.coord == self.planned_path[-1]:
            return True
        return False

    def get_coordinate(self) -> Coordinate:
        return self.coord

    def get_planned_coordinate(self) -> Coordinate:
        if not self.has_completed_task():
            return self.planned_path[self.path_step]

    def __str__(self):
        numb_steps = len(self.planned_path)
        if numb_steps == 0:
            plan = ""
        else:
            plan = f"[{self.planned_path[0]}, ..., {self.planned_path[-1]}]"
        return f"HTOperator(name={self.name}, locked_by={self.locked_by}, path_step={self.path_step}, planned_path=({numb_steps}){plan})"


class HT_Coordinate_View:
    """
    Maintains and monitors the coordinates of HT operators, tracking their positions and deadlock caused by no
    observable movement of HT within a time window.

    Parameters
    ----------
    HT_resource_group : Dict[str, HTOperator]
        A dictionary mapping handling team names to their corresponding HTOperator instances.

    Attributes
    ----------
    __HT_resource_group : Dict[str, HTOperator]
        Internal mapping of HT names to their operator instances.
    __previous_HT_coords : List[Coordinate]
        List of the previous coordinates of all HT, used to monitor movement.
    __no_HT_move_counter : int
        Counter tracking the number of consecutive cycles with no HT movement.
    """

    def __init__(self, HT_resource_group: Dict[str, HTOperator]):
        self.__HT_resource_group: Dict[str, HTOperator] = HT_resource_group
        self.__previous_HT_coords: List[Coordinate] = self.get_all_HT_coordinates()
        self.__no_HT_move_counter: int = 0

    def get_coordinate(self, HT_name: str):
        HT = self.__HT_resource_group.get(HT_name, None)
        if HT:
            return HT.get_coordinate()

    def get_available_HTs(self) -> List[str]:
        available_HTs = list()
        for HT_name, HT_operator in self.__HT_resource_group.items():
            if HT_operator.is_available():
                available_HTs.append(HT_name)
        return available_HTs

    def get_all_HT_coordinates(self):
        coords = list()
        for HT_name in CONSTANT.HT_FLEET.HT_NAMES:
            coords.append(self.get_coordinate(HT_name))
        return coords

    def is_deadlock(self):
        current_HT_coords = self.get_all_HT_coordinates()
        # if there is no HT move from last snapshot
        if current_HT_coords == self.__previous_HT_coords:
            self.__no_HT_move_counter += CONSTANT.JOB_PARAMETER.SYSTEM_TIME_PASSED
            if self.__no_HT_move_counter >= CONSTANT.DEADLOCK_THRESHOLD:
                return True
        else:
            self.__previous_HT_coords = current_HT_coords
            self.__no_HT_move_counter = 0

        return False

    def get_non_moving_HT(self):
        current_HT_coords = self.get_all_HT_coordinates()
        number_of_non_moving_HT = 0
        for prev_coord, current_coord in zip(
            self.__previous_HT_coords, current_HT_coords
        ):
            if prev_coord == current_coord:
                number_of_non_moving_HT += 1

        return number_of_non_moving_HT

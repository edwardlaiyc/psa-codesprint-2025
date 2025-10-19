from enum import Enum
from typing import Any, Dict, List

from src.floor import Coordinate


# Indicate progress status for Job and Resources
class Status(Enum):
    NOT_PLANNED = 0
    NOT_STARTED = 1
    IN_PROGRESS = 2
    COMPLETED = 3


class InstructionType(Enum):
    DRIVE = 0
    BOOK_QC = 1
    BOOK_YARD = 2
    WORK_QC = 3
    WORK_YARD = 24


class JobInstruction:
    """
    Represents an instruction for a job, detailing the type of instruction and associated resources.

    Parameters
    ----------
    instruction_type : InstructionType
        The type/category of the instruction.
    HT_name : str, optional
        The name of HT operator involved (default is None).
    QC_name : str, optional
        The name of QC operator involved (default is None).
    yard_name : str, optional
        The name of yard operator involved (default is None).
    path : List[Coordinate], optional
        A list of coordinates representing the path associated with the DRIVE instruction (default is None).
    start_time : int or None
        The start time of the instruction, to be set later.
    end_time : int or None
        The end time of the instruction, to be set later.
    """

    def __init__(
        self,
        instruction_type: InstructionType,
        HT_name: str = None,
        QC_name: str = None,
        yard_name: str = None,
        path: List[Coordinate] = None,
    ):
        self.instructor_type: InstructionType = instruction_type
        self.HT_name: str = HT_name
        self.QC_name: str = QC_name
        self.yard_name: str = yard_name
        self.path: List[Coordinate] = path
        self.start_time: int = None
        self.end_time: int = None

    def has_started(self) -> bool:
        if self.start_time is not None:
            return True
        return False

    def set_start_time(self, timestamp: int):
        self.start_time = timestamp

    def set_end_time(self, timestamp: int):
        self.end_time = timestamp

    def get_instruction_type(self) -> str:
        return self.instructor_type

    def get_HT_name(self) -> str:
        return self.HT_name

    def get_QC_name(self) -> str:
        return self.QC_name

    def get_yard_name(self) -> str:
        return self.yard_name

    def get_paths(self) -> List[Coordinate]:
        return self.path

    def __str__(self):
        if self.instructor_type == InstructionType.BOOK_QC:
            return f"BOOK_QC({self.QC_name})"
        if self.instructor_type == InstructionType.BOOK_YARD:
            return f"BOOK_YARD({self.yard_name})"
        if self.instructor_type == InstructionType.DRIVE:
            return f"DRIVE({self.HT_name}, Path=({len(self.path)})[{self.path[0]}...{self.path[-1]}])"
        if self.instructor_type == InstructionType.WORK_QC:
            return f"WORK_QC({self.QC_name})"
        if self.instructor_type == InstructionType.WORK_YARD:
            return f"WORK_YARD({self.yard_name})"


class Job:
    """
    Represents a job with associated identifiers, types, statuses, and instructions.

    Attributes
    ----------
    __job_ID : str
        Unique identifier for the job.
    __job_type : str
        Type or category of the job.
    __container_number : str
        Container number related to the job.
    __QC_name : str
        QC involved.
    __QC_job_sequence : str
        QC job sequence identifier.
    __yard_name : str
        Primary candidated yard.
    __alt_yard_names : List[str]
        Alternative yards.
    __assigned_yard_name : str or None
        Yard name currently assigned to the job.
    __assigned_HT_name : str or None
        HT name currently assigned to the job.
    __yard_status : Status
        Involvement status of yard operator.
    __QC_status : Status
        Involvement status of QC operator.
    __HT_status : Status
        Involvment status of HT operator.
    __job_status : Status
        Overall status of the job.
    __instructions : List[JobInstruction]
        List of instructions associated with the job.
    __instruction_stage : int or None
        Current stage of instructions processing.
    __start_time : int or None
        Start time of the job.
    __end_time : int or None
        End time of the job.
    """

    def __init__(
        self,
        job_ID: str,
        job_type: str,
        container_number: str,
        QC_name: str,
        QC_job_sequence: str,
        yard_name: str,
        alt_yard_names: List[str],
    ):
        # job info
        self.__job_ID: str = job_ID
        self.__job_type: str = job_type
        self.__container_number: str = container_number
        self.__QC_name: str = QC_name
        self.__QC_job_sequence: str = QC_job_sequence
        self.__yard_name: str = yard_name
        self.__alt_yard_names: List[str] = alt_yard_names

        # job status
        self.__assigned_yard_name: str = None
        self.__assigned_HT_name: str = None
        self.__yard_status: Status = Status.NOT_PLANNED
        self.__QC_status: Status = Status.NOT_PLANNED
        self.__HT_status: Status = Status.NOT_PLANNED
        self.__job_status: Status = Status.NOT_PLANNED
        self.__instructions: List[JobInstruction] = list()
        self.__instruction_stage: int = None
        self.__start_time: int = None
        self.__end_time: int = None

    def assign_job(self, HT_name: str, yard_name: str):
        self.__assigned_HT_name = HT_name
        self.__assigned_yard_name = yard_name

    def get_job_info(self) -> Dict[str, Any]:
        return {
            "job_ID": self.__job_ID,
            "job_type": self.__job_type,
            "container_number": self.__container_number,
            "QC_name": self.__QC_name,
            "QC_job_sequence": self.__QC_job_sequence,
            "yard_name": self.__yard_name,
            "alt_yard_names": self.__alt_yard_names,
            "assigned_yard_name": self.__assigned_yard_name,
            "assigned_HT_name": self.__assigned_HT_name,
            "yard_status": self.__yard_status,
            "QC_status": self.__QC_status,
            "HT_status": self.__HT_status,
            "job_status": self.__job_status,
            "start_time": self.__start_time,
            "end_time": self.__end_time,
        }

    def set_instructions(self, instructions: List[JobInstruction]):
        self.__instructions = instructions
        self.__instruction_stage = 0
        self.__job_status = Status.NOT_STARTED
        self.__yard_status: Status = Status.NOT_STARTED
        self.__QC_status: Status = Status.NOT_STARTED
        self.__HT_status: Status = Status.NOT_STARTED

    def start_job(self, timestamp: int):
        self.__job_status = Status.IN_PROGRESS
        self.__start_time = timestamp

    def chope_HT(self):
        self.__HT_status = Status.IN_PROGRESS

    def chope_yard(self):
        self.__yard_status = Status.IN_PROGRESS

    def chope_QC(self):
        self.__QC_status = Status.IN_PROGRESS

    def get_latest_instruction(self) -> JobInstruction:
        return self.__instructions[self.__instruction_stage]

    def proceed_to_next_instruction(self, timestamp: int):
        current_instruction = self.get_latest_instruction()
        current_instruction.set_end_time(timestamp)
        instruction_type = current_instruction.get_instruction_type()
        current_instruction_stage = self.__instruction_stage

        # when completed instruction was QC WORK/ YARD WORK, release them
        if current_instruction_stage in [2, 6]:
            if instruction_type == InstructionType.WORK_QC:
                self.__QC_status = Status.COMPLETED
            if instruction_type == InstructionType.WORK_YARD:
                self.__yard_status = Status.COMPLETED

        # when completed instruction is final stage
        if current_instruction_stage >= 7:
            self.__HT_status = Status.COMPLETED
            self.__job_status = Status.COMPLETED
            self.__end_time = timestamp

        # proceed to next stage
        self.__instruction_stage += 1

    def is_HT_required(self):
        return self.__HT_status != Status.COMPLETED

    def is_QC_required(self):
        return self.__QC_status != Status.COMPLETED

    def is_yard_required(self):
        return self.__yard_status != Status.COMPLETED

    def is_completed(self):
        return self.__job_status == Status.COMPLETED

    def __str__(self):
        return f"Job(type={self.__job_type}, job={self.__job_status}, job_seq={self.__QC_job_sequence}, stage=({self.__instruction_stage}){self.__instructions[self.__instruction_stage]}, QC={self.__QC_name}|{self.__QC_status}, HT={self.__assigned_HT_name}|{self.__HT_status}, yard={self.__assigned_yard_name}|{self.__yard_status})"

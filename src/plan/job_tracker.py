from typing import Dict, List, Tuple

import pandas as pd
from logzero import logger

from src.constant import CONSTANT
from src.job import Job, Status


def parse_job_seq(job_seq: str) -> Tuple[str, int]:
    QC_name, seq_number = job_seq.split("_")
    seq_number = int(seq_number)
    return QC_name, seq_number


class JobTracker:
    """
    Tracks and manages jobs parsed from a pandas.DataFrame, maintaining mappings between job sequences and job instances,
    as well as tracking the latest completed job sequence per QC unit.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing job input data to be parsed into Job instances.

    Attributes
    ----------
    job_sequence_map : Dict[str, Job]
        A dictionary mapping job sequence identifiers (strings) to their corresponding Job objects.
    qc_latest_completed_job_seq_map : Dict[str, Optional[str]]
        A dictionary mapping QC unit names to the latest completed job sequence identifier.
        Initialized with None values for each QC name.
    """

    def __init__(self, df: pd.DataFrame):
        self.job_sequence_map: Dict[str, Job] = self.__parse_input_to_jobs(df)
        self.qc_latest_completed_job_seq_map: Dict[str, str] = {
            QC_name: None for QC_name in CONSTANT.QUAY_CRANE_FLOOR.QC_NAMES
        }

    def __parse_input_to_jobs(self, df: pd.DataFrame) -> List[Job]:
        job_sequence_map = dict()
        for index, row in df.iterrows():
            alt_yard_names = []
            if row["ALT_YARD_BLOCK_1"] != "":
                alt_yard_names = [
                    row["ALT_YARD_BLOCK_1"],
                    row["ALT_YARD_BLOCK_2"],
                    row["ALT_YARD_BLOCK_3"],
                ]
            job = Job(
                job_ID=row["JOB_ID"],
                job_type=row["JOB_TYPE"],
                container_number=row["CONTAINER_NO"],
                QC_name=row["QC_M"],
                QC_job_sequence=row["QC_JOB_SEQ"],
                yard_name=row["YARD_BLOCK"],
                alt_yard_names=alt_yard_names,
            )
            job_sequence_map[row["QC_JOB_SEQ"]] = job

        return job_sequence_map

    def fetch_and_update_job_status(self):
        # for each QC, update their plannable jobs to completed (if applicable)
        for QC_name in CONSTANT.QUAY_CRANE_FLOOR.QC_NAMES:
            next_ten_job_seqs = self.get_next_n_job_sequences(
                QC_name=QC_name,
                number_of_jobs=10,
            )
            for job_seq in next_ten_job_seqs:
                job = self.get_job(job_seq)
                job_status = job.get_job_info()["job_status"]
                if (job_status == Status.COMPLETED) and (
                    self.is_next_to_latest_completed_job(job_seq)
                ):
                    self.update_latest_completed_job_seq(QC_name, job_seq)
                    logger.info(
                        f"Updated latest completed job for {QC_name}: {job_seq}"
                    )
                else:
                    break

    def get_next_n_job_sequences(
        self, QC_name: str, number_of_jobs: int = 10
    ) -> List[str]:
        latest_completed_job_seq = self.qc_latest_completed_job_seq_map.get(
            QC_name, None
        )
        last_possible_seq_number = 2500
        # if there is at least one job completed
        if latest_completed_job_seq:
            latest_job_seq = int(latest_completed_job_seq.split("_")[1])
            candidate_seq_numbers = [
                latest_job_seq + i for i in range(1, number_of_jobs + 1, 1)
            ]
            candidate_job_seqs = [
                f"{QC_name}_{seq_number:04d}"
                for seq_number in candidate_seq_numbers
                if seq_number <= last_possible_seq_number
            ]

        # no job completed yet
        else:
            candidate_job_seqs = [
                f"{QC_name}_{i:04d}" for i in range(1, number_of_jobs + 1, 1)
            ]
        return candidate_job_seqs

    def update_latest_completed_job_seq(self, QC_name: str, job_seq: str):
        if QC_name not in CONSTANT.QUAY_CRANE_FLOOR.QC_NAMES:
            raise ValueError(f"{QC_name} not in QC list.")

        self.qc_latest_completed_job_seq_map[QC_name] = job_seq

    def is_next_to_latest_completed_job(self, job_seq: str) -> bool:
        QC_name, seq_number = parse_job_seq(job_seq)
        latest_completed_job_seq = self.qc_latest_completed_job_seq_map.get(
            QC_name, None
        )
        if latest_completed_job_seq:
            _, latest_seq_number = parse_job_seq(latest_completed_job_seq)
            if latest_seq_number + 1 == seq_number:
                return True
            else:
                return False
        return seq_number == 1

    def get_plannable_job_sequences(self):
        # get QC jobs seq that is 10-step ahead from current completed one
        ahead_job_seqs = list()
        for QC_name in CONSTANT.QUAY_CRANE_FLOOR.QC_NAMES:
            next_ten_jobs = self.get_next_n_job_sequences(
                QC_name=QC_name,
                number_of_jobs=10,
            )
            ahead_job_seqs.extend(next_ten_jobs)

        # filter out those have been planned (in previous iteration)
        plannable_job_seqs = list()
        for job_seq in ahead_job_seqs:
            job = self.get_job(job_seq)
            job_info = job.get_job_info()
            if job_info["job_status"] == Status.NOT_PLANNED:
                plannable_job_seqs.append(job_seq)

        return plannable_job_seqs

    def is_all_job_completed(self):
        for (
            QC_name,
            latest_completed_job_seq,
        ) in self.qc_latest_completed_job_seq_map.items():
            last_possible_job_seq = f"{QC_name}_2500"
            if last_possible_job_seq != latest_completed_job_seq:
                return False

        return True

    def get_number_of_completed_jobs(self) -> Dict[str, int]:
        total_completed_jobs = 0
        for (
            QC_name,
            latest_completed_job_seq,
        ) in self.qc_latest_completed_job_seq_map.items():
            # when no job completed yet
            if latest_completed_job_seq is None:
                continue
            # when at least one job completed
            else:
                # here we only take latest completed job that follow SEQUENCE
                # it could be higher, but I concluded the mismatch is negligible
                latest_job_seq = int(latest_completed_job_seq.split("_")[1])
                total_completed_jobs += latest_job_seq

        return total_completed_jobs

    def get_job(self, job_seq: str):
        return self.job_sequence_map.get(job_seq, None)

    def export_job_report(self):
        data = list()
        for job_seq, job in self.job_sequence_map.items():
            data.append(job.get_job_info())

        return pd.DataFrame(data=data)

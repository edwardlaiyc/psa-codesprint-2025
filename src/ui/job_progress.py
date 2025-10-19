from typing import Dict


class JobProgressDisplay:
    """
    Displays and tracks the progress and status of various job-related entities in the terminal.

    Attributes
    ----------
    data : dict
        A nested dictionary containing job progress metrics and resource statuses, including:

        - 'JOBS': Dictionary with total jobs, remaining jobs, completed jobs, and elapsed time in seconds.
        - 'QUAY CRANE': Dictionary with total quay cranes, active and idle counts.
        - 'HT': Dictionary with total HT, active and idle counts.
        - 'YARD': Dictionary with total yards, active and idle counts.
        - 'HT LOCATIONS': List of HT location labels and their coordinates.

    """

    def __init__(self):
        self.data = {
            "JOBS": {
                "TOTAL": 20000,
                "REMAINING": 20000,
                "COMPLETED": 0,
                "TIME(secs)": 0,
            },
            "QUAY CRANE": {
                "TOTAL": 8,
                "ACTIVE": 0,
                "IDLE": 8,
            },
            "HT": {
                "TOTAL": 80,
                "MOVING": 0,
                "NOT MOVING": 80,
            },
            "YARD": {"TOTAL": 16, "ACTIVE": 0, "IDLE": 16},
            "HT LOCATIONS": [[f"HT_{i:02}", "(XX,YY)"] for i in range(1, 81, 1)],
        }

    def update_data(self, data: Dict):
        self.data = data

    def update_time(self, time: int):
        self.data["JOBS"]["TIME(secs)"] = time

    def get_time(self):
        return self.data["JOBS"]["TIME(secs)"]

    def get_data(self):
        return self.data

    def get_job_status(self):
        status = [f" {attr:<10}:{val:>9,} " for attr, val in self.data["JOBS"].items()]
        return "\n".join(status)

    def get_quay_crane_status(self):
        status = [
            f" {attr:<7}:{val:>4,} " for attr, val in self.data["QUAY CRANE"].items()
        ] + [
            ""
        ]  # space padding

        return "\n".join(status)

    def get_HT_status(self):
        status = [
            f" {attr:<10}:{val:>3,} " for attr, val in self.data["HT"].items()
        ] + [
            ""
        ]  # space padding

        return "\n".join(status)

    def get_yard_status(self):
        status = [
            f" {attr:<7}:{val:>4,} " for attr, val in self.data["YARD"].items()
        ] + [
            ""
        ]  # space padding

        return "\n".join(status)

    def get_HT_locations(self):
        # display the first 36 HTs
        HT_locations = self.data["HT LOCATIONS"]
        formatted_locations = [f"{i[0]}{i[1]}" for i in HT_locations]
        status = ""
        start_id = 0
        last_id = 36
        step = 6
        for idx in range(start_id, last_id, step):
            line = " | ".join(formatted_locations[idx : idx + step])
            status += line + "\n"
        return status

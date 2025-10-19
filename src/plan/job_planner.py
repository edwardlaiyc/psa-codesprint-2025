from typing import List

from logzero import logger

from src.constant import CONSTANT
from src.floor import Coordinate, SectorMapSnapshot
from src.job import InstructionType, Job, JobInstruction
from src.operators import HT_Coordinate_View
from src.plan.job_tracker import JobTracker


class JobPlanner:
    """
    Coordinates job planning activities using HT tracker and sector map data.

    Attributes
    ----------
    ht_coord_tracker : HT_Coordinate_View
        An instance responsible for tracking the coordinates of HTs.
    sector_map_snapshot : SectorMapSnapshot
        A snapshot of the sector map representing the current state of the environment for planning.
    """

    def __init__(
        self,
        ht_coord_tracker: HT_Coordinate_View,
        sector_map_snapshot: SectorMapSnapshot,
    ):
        self.ht_coord_tracker = ht_coord_tracker
        self.sector_map_snapshot = sector_map_snapshot

    def is_deadlock(self):
        return self.ht_coord_tracker.is_deadlock()

    def get_non_moving_HT(self):
        return self.ht_coord_tracker.get_non_moving_HT()

    """ YOUR TASK HERE
    Objective: modify the following functions (including input arguments as you see fit) to achieve better planning efficiency.
        select_HT():
            select HT for the job based on your self-defined logic.
        select_yard():
            select yard for the job based on your self-defined logic.
        get_path_from_buffer_to_QC():
        get_path_from_buffer_to_yard():
        get_path_from_yard_to_buffer():
        get_path_from_QC_to_buffer():
            generate an efficient path for HT to navigate between listed locations (QC, yard, buffer).        
    """

    def plan(self, job_tracker: JobTracker) -> List[Job]:
        # logger.info("Planning started.")
        plannable_job_seqs = job_tracker.get_plannable_job_sequences()
        selected_HT_names = list()  # avoid selecting duplicated HT during the process
        new_jobs = list()  # container for newly created jobs

        # create job loop: ranging from 0 to at most 16 jobs
        for job_seq in plannable_job_seqs:
            # parse job info
            job = job_tracker.get_job(job_seq)
            job_info = job.get_job_info()
            job_type, QC_name, yard_name, alt_yard_names = [
                job_info[k]
                for k in ["job_type", "QC_name", "yard_name", "alt_yard_names"]
            ]

            # select HT for the job based on job type, return None if no HT available or applicable
          # HT_name = self.select_HT1(job_type, selected_HT_names, QC_name, yard_name)
            HT_name = self.select_HT2(job_type, selected_HT_names, QC_name, yard_name)

            # not proceed with job planning if no available HTs
            if HT_name is None:
                break
            selected_HT_names.append(HT_name)

            # select yard if the job is DISCHARGE
            if job_type == CONSTANT.JOB_PARAMETER.DISCHARGE_JOB_TYPE:
                yard_name = self.select_yard(yard_name, alt_yard_names)

            # record the assigned HT and yard
            job.assign_job(HT_name=HT_name, yard_name=yard_name)

            # construct the job instructions
            job_instructions = list()
            buffer_coord = self.ht_coord_tracker.get_coordinate(HT_name)

            # For DI job
            if job_type == CONSTANT.JOB_PARAMETER.DISCHARGE_JOB_TYPE:

                # 1. Book QC resource
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.BOOK_QC,
                    )
                )

                # 2. HT drives from Buffer to QC[IN]
                buffer_coord = self.ht_coord_tracker.get_coordinate(HT_name)
                path = self.get_path_from_buffer_to_QC(buffer_coord, QC_name)
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.DRIVE,
                        HT_name=HT_name,
                        path=path,
                    )
                )

                # 3. Work with QC
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.WORK_QC,
                        HT_name=HT_name,
                        QC_name=QC_name,
                    )
                )

                # 4. HT drives from QC to Buffer
                path = self.get_path_from_QC_to_buffer(QC_name, buffer_coord)
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.DRIVE,
                        HT_name=HT_name,
                        path=path,
                    )
                )

                # 5. Book Yard resource
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.BOOK_YARD,
                    )
                )

                # 6. HT drives from Buffer to Yard[IN]
                path = self.get_path_from_buffer_to_yard(buffer_coord, yard_name)
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.DRIVE,
                        HT_name=HT_name,
                        path=path,
                    )
                )

                # 7. Work with Yard
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.WORK_YARD,
                        HT_name=HT_name,
                        yard_name=yard_name,
                    )
                )

                # 8. HT drives from Yard to Buffer
                path = self.get_path_from_yard_to_buffer(yard_name, buffer_coord)
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.DRIVE,
                        HT_name=HT_name,
                        path=path,
                    )
                )

            # For LO job
            else:

                # 1. Book Yard resource
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.BOOK_YARD,
                    )
                )

                # 2. HT drives from buffer to Yard[IN]
                buffer_coord = self.ht_coord_tracker.get_coordinate(HT_name)
                path = self.get_path_from_buffer_to_yard(buffer_coord, yard_name)
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.DRIVE,
                        HT_name=HT_name,
                        path=path,
                    )
                )

                # 3. Work with Yard
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.WORK_YARD,
                        HT_name=HT_name,
                        yard_name=yard_name,
                    )
                )

                # 4. HT drives from Yard to buffer
                path = self.get_path_from_yard_to_buffer(yard_name, buffer_coord)
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.DRIVE,
                        HT_name=HT_name,
                        path=path,
                    )
                )

                # 5. Book QC resource
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.BOOK_QC,
                    )
                )

                # 6. HT drives from buffer to QC[IN]
                path = self.get_path_from_buffer_to_QC(buffer_coord, QC_name)
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.DRIVE,
                        HT_name=HT_name,
                        path=path,
                    )
                )

                # 7. Work with QC
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.WORK_QC,
                        HT_name=HT_name,
                        QC_name=QC_name,
                    )
                )

                # 8. HT drives from QC to buffer
                path = self.get_path_from_QC_to_buffer(QC_name, buffer_coord)
                job_instructions.append(
                    JobInstruction(
                        instruction_type=InstructionType.DRIVE,
                        HT_name=HT_name,
                        path=path,
                    )
                )

            job.set_instructions(job_instructions)
            new_jobs.append(job)
            # logger.debug(f"{job}")

        return new_jobs

    # HT ASSIGNMENT LOGIC
    def select_HT1(self, job_type: str, selected_HT_names: List[str], QC_name: str, yard_name: str) -> str:
        """
        Selects an available HT (Horizontal Transport) based on the job type and a list of already selected HTs.

        OLD LOGIC:
        For a discharge job, the method selects the first unselected HT from the left (start) of the buffer zone.
        For any other job type, it selects the first unselected HT from the right (end) of the buffer zone.

        NEW LOGIC:
        #finds HT that is closest to the QC/yard to minimise horizontal travel


        Args:
            job_type (str): The type of job to be processed (e.g., discharge or other).
            selected_HT_names (List[str]): A list of HT names that are already selected or in use.
            QC_name (str): Name of QC to go to
            yard_name (str): Name of yard to go to

        Returns:
            str or None: The name of the selected HT if one is available; otherwise, None.
        """
        plannable_HTs = self.ht_coord_tracker.get_available_HTs()
        selected_HT = None
        # get coordinates of QC_in and yard_in
        QC_in_coord = self.sector_map_snapshot.get_QC_sector(QC_name).in_coord
        yard_in_coord = self.sector_map_snapshot.get_yard_sector(yard_name).in_coord

        # if DI job, track the closest HT to QC_in
        if job_type == CONSTANT.JOB_PARAMETER.DISCHARGE_JOB_TYPE:
            max_x = 1
            for HT_name in plannable_HTs:
                if HT_name not in selected_HT_names:
                    buffer_coord = self.ht_coord_tracker.get_coordinate(HT_name)
                    selected_HT = HT_name
                    # as long as HT is still left of QC, move right until
                    # we are just to the right of QC
                    # we then have the HT closest to the QC
                    if buffer_coord.x < QC_in_coord.x:
                        if buffer_coord.x > max_x:
                            selected_HT = HT_name
                            max_x = buffer_coord.x
            
        # if LO job, track closest HT to yard_in
        else:
            max_x = 1
            for HT_name in plannable_HTs[::-1]:
                if HT_name not in selected_HT_names:
                    buffer_coord = self.ht_coord_tracker.get_coordinate(HT_name)
                    selected_HT = HT_name
                    # similarly, iterate right until we find the HT
                    # just to the right of the yard
                    if buffer_coord.x < yard_in_coord.x:
                        if buffer_coord.x > max_x:
                            selected_HT = HT_name
                            max_x = buffer_coord.x
        return selected_HT
    
    def select_HT2(self, job_type: str, selected_HT_names: List[str], QC_name: str, yard_name: str) -> str:
        """
        Selects an available HT (Horizontal Transport) based on the job type and a list of already selected HTs.
        
        OLD LOGIC:
        For a discharge job, the method selects the first unselected HT from the left (start) of the buffer zone.
        For any other job type, it selects the first unselected HT from the right (end) of the buffer zone.

        NEW LOGIC:
        This version finds the leftmost HT available
        Chosen HT can go straight up, and right, to the QC
        Gives the best timing currently

        Args:
            job_type (str): The type of job to be processed (e.g., discharge or other).
            selected_HT_names (List[str]): A list of HT names that are already selected or in use.
            QC_name (str): Name of QC to go to
            yard_name (str): Name of yard to go to

        Returns:
            str or None: The name of the selected HT if one is available; otherwise, None.
        """
        plannable_HTs = self.ht_coord_tracker.get_available_HTs()
        selected_HT = None
        QC_in_coord = self.sector_map_snapshot.get_QC_sector(QC_name).in_coord
        yard_in_coord = self.sector_map_snapshot.get_yard_sector(yard_name).in_coord

        # if DI job, pick the HT on far left of buffer zone
        if job_type == CONSTANT.JOB_PARAMETER.DISCHARGE_JOB_TYPE:
            max_x = 1
            for HT_name in plannable_HTs:
                if HT_name not in selected_HT_names:
                    buffer_coord = self.ht_coord_tracker.get_coordinate(HT_name)
                    selected_HT = HT_name
                    #the first HT that is to the left of the QC, use that HT
                    if buffer_coord.x < QC_in_coord.x:
                        break
                        if buffer_coord.x > max_x:
                            selected_HT = HT_name
                            max_x = buffer_coord.x
            
        # otherwise far right
        else:
            max_x = 1
            for HT_name in plannable_HTs[::-1]:
                if HT_name not in selected_HT_names:
                    buffer_coord = self.ht_coord_tracker.get_coordinate(HT_name)
                    selected_HT = HT_name
                    #the first HT that is left of the yard, use that HT
                    if buffer_coord.x < yard_in_coord.x:
                        break
                        if buffer_coord.x > max_x:
                            selected_HT = HT_name
                            max_x = buffer_coord.x
        return selected_HT

    # YARD ASSIGNMENT LOGIC:
    def select_yard(self, yard_name: str, alt_yard_name: List[str]) -> str:
        
        """
        OLD LOGIC:
        Selects a yard for use. Currently, simply returns the provided yard name.

        NEW LOGIC:
        Select yard based on the one with the lowest current count.
        To ensure even distribution, reduce congestion, ensure the 700 limit not exceeded.

        Args:
            yard_name (str): The name of the yard to select.
            alt_yard_name (str): Name of alternative yards to select.

        Returns:
            str: The selected yard name.
        """
        # create dict to store count for all yards
        if not hasattr(self, 'yard_DI_job_count'):
            self.yard_DI_job_count = {}

        # list of yards to select from
        yard_list = [yard_name] + alt_yard_name

        # create entry in dict for each yard if does not already exist
        for yard in yard_list:
            if yard not in self.yard_DI_job_count:
                self.yard_DI_job_count[yard]  = 0
        
        # create sorted list of (yard, count) from lowest to highest count
        yard_job_count = {key: value for key, value in self.yard_DI_job_count.items() if key in yard_list }
        yard_job_count = sorted(yard_job_count.items(), key=lambda item: item[1])

        # select the yard with the lowest count
        selected_yard = yard_job_count[0][0]

        #  the count of that yard
        self.yard_DI_job_count[selected_yard] += 1

        # print if count of selected yard exceeds 700
        if self.yard_DI_job_count[selected_yard] >= 700:
            print(self.yard_DI_job_count[selected_yard])
        return selected_yard

    # NAVIGATION LOGIC
    def get_path_from_buffer_to_QC(
        self, buffer_coord: Coordinate, QC_name: str
    ) -> List[Coordinate]:
        """
        Generates a path from a buffer location to a Quay Crane (QC) input coordinate.

        OLD LOGIC:
        The path follows a predefined route:
        1. Moves south to the highway left lane (y = 7).
        2. Travels west along the highway to the left boundary (x = 1).
        3. Moves north to the upper lane (y = 4).
        4. Travels east to the IN coordinate of the specified QC.

        NEW LOGIC:
        If HT is on the right of the QC,
        1. Moves south to the highway left lane (y = 7).
        2. Travels west along the highway to the left boundary (x = 1).
        3. Moves north to the upper lane (y = 4).
        4. Travels east to the IN coordinate of the specified QC.

        If HT is on the left of the QC,
        1. Moves straight up from buffer to lane 4
        2. Travels east to the IN coordinate of the specified QC.

        Args:
            buffer_coord (Coordinate): The starting coordinate in the buffer zone.
            QC_name (str): The name of the Quay Crane to which the path should lead.

        Returns:
            List[Coordinate]: A list of coordinates representing the path from the buffer to the QC.
        """
        QC_in_coord = self.sector_map_snapshot.get_QC_sector(QC_name).in_coord
        
        # HT on the right of QC
        if buffer_coord.x > QC_in_coord.x:
            # go South to take Highway Left lane (y=7)
            highway_lane_y = 7
            path = [Coordinate(buffer_coord.x, highway_lane_y)]

            # then go to the left boundary
            path.extend(
                [Coordinate(x, highway_lane_y) for x in range(buffer_coord.x - 1, 0, -1)]
            )

            # then go to upper boundary and navigate to QC_in
            up_path_x = 1
            path.extend([Coordinate(up_path_x, y) for y in range(6, 3, -1)])
            qc_travel_lane_y = 4
            path.extend(
                [Coordinate(x, qc_travel_lane_y) for x in range(2, QC_in_coord.x + 1, 1)]
            )
        
        # HT on the left of QC
        else:
            # go straight up to upper boundary
            up_path_x = buffer_coord.x
            path = []
            path.extend([Coordinate(up_path_x, y) for y in range(5, 3, -1)])
            
            # navigate to QC_in
            qc_travel_lane_y = 4
            path.extend(
                [Coordinate(x, qc_travel_lane_y) for x in range(up_path_x + 1, QC_in_coord.x + 1, 1)]
            )

        path.append(QC_in_coord)

        return path

    def get_path_from_buffer_to_yard(
        self, buffer_coord: Coordinate, yard_name: str
    ) -> List[Coordinate]:
        """
        Generates a path from a buffer location to a yard IN area's coordinate.

        OLD LOGIC:
        The path follows a specific route:
        1. Moves north to the QC travel lane (y = 5).
        2. Travels east to the right boundary of the sector (x = 42).
        3. Moves south to the Highway Left lane (y = 11).
        4. Travels west along the highway to the left boundary (x = 1).
        5. Moves south to the lower boundary (y = 12).
        6. Travels east to the IN coordinate of the specified yard.

        NEW LOGIC:
        *only move south on even-numbered lanes

        If HT to the right of yard:
        1. Go south to highway 7
        2. Move left to down_path (right above yard_in or one to the left)
        3. Move south to highway 12
        4. navigate to yard_in

        If HT to the left of yard:
        1. Move down using closest even numbered path on the left of the HT
        2. Move to the down path
        3. Travel right to the IN coordinate of the specified yard

        Args:
            buffer_coord (Coordinate): The starting coordinate in the buffer zone.
            yard_name (str): The name of the yard to which the path should lead.

        Returns:
            List[Coordinate]: A list of coordinates representing the path from the buffer to the yard.
        """
        yard_in_coord = self.sector_map_snapshot.get_yard_sector(yard_name).in_coord
     
        # HT on the right of yard
        if buffer_coord.x  >= yard_in_coord.x:
            down_path_x = yard_in_coord.x #HT will go down at yard.x
            if down_path_x % 2 != 0: # if odd change to even
                down_path_x -= 1
            # Go South to take highway lane (y=7), then go to down_path
            path = [Coordinate(buffer_coord.x, buffer_coord.y + 1)]
            highway_lane_y = 7
            path.extend(
                [Coordinate(x, highway_lane_y) for x in range(buffer_coord.x - 1, down_path_x -1, -1)]
            )

            # go down to Highway lane(12)
            path.extend([Coordinate(down_path_x, y) for y in range(8, 13, 1)])
            highway_lane_y = 12
            # navigate to yard_in
            path.extend([Coordinate(x, highway_lane_y) for x in range(down_path_x + 1, yard_in_coord.x + 1, 1)])

        # HT on the left of yard
        else:
            down_path_x = buffer_coord.x #HT will go down at current x-coord
            if down_path_x % 2 != 0: #if odd change to even
                down_path_x -= 1
            # shift down to highway 7, adjust to even number lane
            path = [Coordinate(x, 7) for x in range(buffer_coord.x, down_path_x -1, -1)]
            # go down to lane 12 using even lane
            path.extend([Coordinate(down_path_x, y) for y in range(8, 13, 1)])
            # navigate right to yard_in
            highway_lane_y = 12
            path.extend([Coordinate(x, highway_lane_y) for x in range(down_path_x + 1, yard_in_coord.x + 1, 1)])

        path.append(yard_in_coord)
        return path

    def get_path_from_yard_to_buffer(
        self, yard_name: str, buffer_coord: Coordinate
    ) -> List[Coordinate]:
        """
        Generates a path from a yard OUT area's coordinate to a buffer location.

        OLD LOGIC:
        The path follows this route:
        1. Starts at the yard OUT coordinate.
        2. Moves east along the highway lane (y = 12) towards the second-to-right boundary.
        3. Moves north to the Highway Left lane (y = 7).
        4. Travels west along the highway left lane to the target buffer coordinate.

        NEW LOGIC:
        * only move up on odd numbered lanes.

        If the HT location in the buffer is on the right or same x coordinate of the yard:
        1. Starts at the yard OUT coordinate.
        2. select the odd number vertical path on the right of the buffer
        3. move right along Lane 12 to the x coordinate of the buffer
        4. move up the selected odd number path
        5. move left along Lane 7 to the buffer if needed

        If HT location of on the left of the yard
        1. Starts at the yard OUT coordinate.
        2. select the odd number vertical path on the right of the buffer
        3. move to the right along Lane 12 to the vertical path
        4. move up the vertical path
        5. move left along lane 7 to the buffer

        Args:
            yard_name (str): The name of the yard from which the path starts.
            buffer_coord (Coordinate): The destination coordinate in the buffer zone.

        Returns:
            List[Coordinate]: A list of coordinates representing the path from the yard to the buffer.
        """
        yard_out_coord = self.sector_map_snapshot.get_yard_sector(yard_name).out_coord

        # go to Yard[OUT] first
        path = [yard_out_coord]

        # enter highway lane, go to tile second-to-right boundary
        highway_lane_y = 12
        # if HT location in the buffer is on the right or same x coordinate of the yard
        if buffer_coord.x >= yard_out_coord.x:
            # select the odd number vertical path on the right of the buffer
            up_path_x = buffer_coord.x
            if up_path_x % 2 == 0:  # if even, change to odd
                up_path_x += 1
            # move right along Lane 12 to the x coordinate of the buffer
            path.extend(
                [Coordinate(x, highway_lane_y) for x in range(yard_out_coord.x, up_path_x + 1, 1)]
            )
            # move up the selected odd number path 
            path.extend([Coordinate(up_path_x, y) for y in range(11, 6, -1)])
            
            # move left along Lane 7 to the buffer if needed
            if up_path_x > buffer_coord.x:
                path.extend([Coordinate(x, 7) for x in range(up_path_x - 1, buffer_coord.x - 1, -1)])
        
        # buffer on the left of the yard
        else:
            # select the odd number vertical path on the right of the buffer
            up_path_x = yard_out_coord.x
            if up_path_x % 2 == 0: # if even, change to odd
                up_path_x += 1

            # move to the right along Lane 12 to the vertical path
            path.extend(Coordinate(x, highway_lane_y) for x in range(yard_out_coord.x, up_path_x + 1, 1))
            # move up the vertical path
            path.extend([Coordinate(up_path_x, y) for y in range(11, 6, -1)])
            # move left along lane 7 to the buffer
            path.extend([Coordinate(x, 7) for x in range(up_path_x - 1, buffer_coord.x - 1, -1)])
        
        path.append(buffer_coord)
        return path

    def get_path_from_QC_to_buffer(
        self, QC_name: str, buffer_coord: Coordinate
    ) -> List[Coordinate]:
        """
        Generates a path from a Quay Crane (QC) OUT coordinate to a buffer location.

        The path follows this route:
        1. Starts at the QC OUT coordinate.
        2. Moves south to the QC travel lane (y = 4).
        3. Travels east along the QC travel lane to the right boundary.
        4. Moves south to the Highway Left lane (y = 7).
        5. Travels west along the highway left lane to the buffer coordinate.

        Args:
            QC_name (str): The name of the Quay Crane from which the path starts.
            buffer_coord (Coordinate): The destination coordinate in the buffer zone.

        Returns:
            List[Coordinate]: A list of coordinates representing the path from the QC to the buffer.
        """
        QC_out_coord = self.sector_map_snapshot.get_QC_sector(QC_name).out_coord

        # go to QC_out first
        path = [QC_out_coord]
            
        # go South to take QC Travel Lane
        qc_travel_lane_y = 4
        path.append(Coordinate(QC_out_coord.x, qc_travel_lane_y))
        # move all the way to right boundary
        path.extend(
            [Coordinate(x, qc_travel_lane_y) for x in range(QC_out_coord.x + 1, 43, 1)]
        )

        # go down to Highway Left lane(7), then takes left most
        down_path_x = 42
        path.extend([Coordinate(down_path_x, y) for y in range(5, 8, 1)])

        # navigate back to buffer
        highway_lane_y = 7
        path.extend(
            [Coordinate(x, highway_lane_y) for x in range(41, buffer_coord.x - 1, -1)]
        )
        path.append(buffer_coord)

        return path

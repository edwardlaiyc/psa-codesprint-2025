from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.constant import CONSTANT


# Datastructure to capture 2-D Coordinate in Terminal Map
@dataclass
class Coordinate:
    x: int
    y: int

    def __str__(self):
        return f"({self.x:02d},{self.y:02d})"


# Datastructure to capture 2-D Coordinate for entity that has different locations for IN and OUT gate.
# Example: QC and yard.
@dataclass
class In_Out_Coord:
    in_coord: Coordinate
    out_coord: Coordinate


class Sector:
    """Represents a sector within the terminal, defined by a coordinate and possible movement directions.

    Parameters
    ----------
    coordinate : Coordinate
        The coordinate location of the sector.
    moveable_directions : List[str], optional
        A list of directions in which movement from this sector is allowed.
        Defaults to ["←", "↑", "→", "↓"].
    capacity : int, optional
        The maximum number of occupators this sector can hold. Defaults to 1.

    Attributes
    ----------
    __coordinate : Coordinate
        The coordinate of the sector.
    __moveable_coordinates : List[Coordinate]
        Coordinates to which movement is possible from this sector.
    __occupators : List[str]
        List of identifiers for current occupators of this sector.
    __capacity : int
        Maximum number of occupators allowed in this sector.
    """

    def __init__(
        self,
        coordinate: Coordinate,
        moveable_directions: List[str] = list("←↑→↓"),
        capacity: int = 1,
    ):
        self.__coordinate: Coordinate = coordinate
        self.__moveable_coordinates: List[Coordinate] = (
            self.__get_onscreen_movable_coordinates(moveable_directions)
        )

        self.__occupators: List[str] = list()
        self.__capacity: int = capacity

    def __get_onscreen_movable_coordinates(
        self, moveable_directions: List[str]
    ) -> List[Coordinate]:
        """Supporting to set up movable coordinates based on customized directions,
        with additional filtering if the coordinate is out of UI screen.
        """
        onscreen_coordinates = list()
        for direction in moveable_directions:
            # based on direction to determine coordinate can move to
            if direction == "←":
                next_x, next_y = self.__coordinate.x - 1, self.__coordinate.y
            elif direction == "→":
                next_x, next_y = self.__coordinate.x + 1, self.__coordinate.y
            elif direction == "↑":
                next_x, next_y = self.__coordinate.x, self.__coordinate.y - 1
            elif direction == "↓":
                next_x, next_y = self.__coordinate.x, self.__coordinate.y + 1

            # validate the sector is not out of the map
            if (1 <= next_x <= 42) and (3 <= next_y <= 13):
                # if y = 3, they belongs to QC's IN and OUT
                QC_valid_x = [i + j for i in range(3, 42, 5) for j in range(2)]
                if (next_y == 3) and (next_x not in QC_valid_x):
                    continue

                # if y = 13, they belongs to Yard's IN and OUT
                yard_valid_x = [i + j for i in range(2, 42, 5) for j in range(4)]
                if (next_y == 13) and (next_x not in yard_valid_x):
                    continue

                onscreen_coordinates.append(Coordinate(next_x, next_y))

        return onscreen_coordinates

    def get_movable_to_coordinates(self) -> List[Coordinate]:
        return self.__moveable_coordinates

    def is_available(self) -> bool:
        return len(self.__occupators) < self.__capacity

    def get_occupators(self) -> List[str]:
        return self.__occupators.copy()

    def get_capacity(self) -> int:
        return self.__capacity

    def get_coordinate(self) -> Coordinate:
        return self.__coordinate

    def add_occupator(self, HT_name: str):
        if HT_name in self.__occupators:
            raise ValueError(f"{HT_name} is at {self.__coordinate} already.")
        self.__occupators.append(HT_name)

    def remove_occupator(self, HT_name: str):
        number_of_occupators = len(self.__occupators)
        if number_of_occupators <= 0:
            raise ValueError(f"There is no occupator at {self.__coordinate} to remove.")
        self.__occupators.remove(HT_name)

    def has_occupator(self, HT_name: str) -> bool:
        return HT_name in self.__occupators

    def __str__(self):
        return f"<{self.__class__.__name__}: {self.__coordinate}>"


class Unused_Sector(Sector):
    def __init__(self, coordinate):
        super().__init__(coordinate=coordinate, moveable_directions=[""], capacity=0)


class QC_IN_Sector(Sector):
    def __init__(self, coordinate):
        super().__init__(coordinate=coordinate, moveable_directions=["→"], capacity=2)


class QC_OUT_Sector(Sector):
    def __init__(self, coordinate):
        super().__init__(coordinate=coordinate, moveable_directions=["↓"], capacity=2)


class QC_Travel_Sector(Sector):
    def __init__(self, coordinate):
        super().__init__(
            coordinate=coordinate, moveable_directions=list("↑↓→"), capacity=1
        )


class Buffer_Zone_Sector(Sector):
    def __init__(self, coordinate):
        super().__init__(
            coordinate=coordinate, moveable_directions=list("↑↓"), capacity=2
        )


class Highway_Right_Sector(Sector):
    def __init__(self, coordinate):
        super().__init__(
            coordinate=coordinate, moveable_directions=list("↑↓→"), capacity=1
        )


class Highway_Left_Sector(Sector):
    def __init__(self, coordinate):
        super().__init__(
            coordinate=coordinate, moveable_directions=list("↑↓←"), capacity=1
        )


class Yard_IN_Sector(Sector):
    def __init__(self, coordinate):
        super().__init__(coordinate=coordinate, moveable_directions=["→"], capacity=3)


class Yard_OUT_Sector(Sector):
    def __init__(self, coordinate):
        super().__init__(coordinate=coordinate, moveable_directions=["↑"], capacity=1)


# initialize SectorMap
class SectorFactory:
    def __init__(self):
        pass

    def create_sector(self, sector_type: str, coordinate: Coordinate):
        if sector_type == "Unused":
            return Unused_Sector(coordinate)
        elif sector_type == "QC_IN":
            return QC_IN_Sector(coordinate)
        elif sector_type == "QC_OUT":
            return QC_OUT_Sector(coordinate)
        elif sector_type == "QC_TRAVEL":
            return QC_Travel_Sector(coordinate)
        elif sector_type == "BUFFER":
            return Buffer_Zone_Sector(coordinate)
        elif sector_type == "HIGHWAY_RIGHT":
            return Highway_Right_Sector(coordinate)
        elif sector_type == "HIGHWAY_LEFT":
            return Highway_Left_Sector(coordinate)
        elif sector_type == "YARD_IN":
            return Yard_IN_Sector(coordinate)
        elif sector_type == "YARD_OUT":
            return Yard_OUT_Sector(coordinate)
        else:
            raise ValueError(f"The sector_type is invalid: {sector_type}")


class SectorMap:
    def __init__(self):
        self.__data = self.__initialize_data()
        self.__QC_sector_map = self.__set_QC_sector_map()
        self.__buffer_sector_coords = self.__set_buffer_sector_coords()
        self.__yard_sector_map = self.__set_yard_sector_map()

    def __initialize_data(self):
        sector_type_map = list()

        # first 3 rows
        sector_factory = SectorFactory()
        sector_type_map.extend([[None] * CONSTANT.SCREEN.NUMBER_OF_SECTORS_X] * 3)

        # 1 QC in-out
        qc_sectors = [None] * 3 + ["QC_IN", "QC_OUT", None, None, None] * 8
        sector_type_map.append(qc_sectors)

        # 2 QC Travel lane
        qc_travel_lane_sectors = [None] + ["QC_TRAVEL"] * 42
        sector_type_map.extend([qc_travel_lane_sectors] * 2)

        # 1 Bufer lane
        buffer_lane_sectors = [None] + ["BUFFER"] * 42
        sector_type_map.append(buffer_lane_sectors)

        # 6 Highway lane
        highway_left_sectors = [None] + ["HIGHWAY_LEFT"] * 42
        highway_right_sectors = [None] + ["HIGHWAY_RIGHT"] * 42
        sector_type_map.extend(
            [
                highway_left_sectors,
                highway_right_sectors,
                highway_left_sectors,
                highway_right_sectors,
                highway_left_sectors,
                highway_right_sectors,
            ]
        )

        # 1 Yard in-out
        yard_sectors = (
            [None] * 2
            + ["YARD_IN", "YARD_OUT", "YARD_IN", "YARD_OUT", None] * 8
            + [None]
        )
        sector_type_map.append(yard_sectors)

        # Follow sector type to create corresponding sectors
        sector_map = list()
        sector_type_map = np.array(sector_type_map)
        min_x, max_x = CONSTANT.COORDINATE_MAP.X_RANGE
        min_y, max_y = CONSTANT.COORDINATE_MAP.Y_RANGE
        for row_id in range(min_y, max_y + 1, 1):
            sector_row = list()
            for col_id in range(min_x, max_x + 1, 1):
                sector_type = sector_type_map[row_id][col_id]
                # if this is a sector
                if sector_type:
                    sector_row.append(
                        sector_factory.create_sector(
                            sector_type, Coordinate(col_id, row_id)
                        )
                    )
                # otherwise padding with None
                else:
                    sector_row.append(None)
            sector_map.append(sector_row)
        sector_map = np.array(sector_map)
        return sector_map

    def __set_QC_sector_map(self) -> Dict[str, In_Out_Coord]:
        QC_sector_map = {
            f"QC{i+1}": In_Out_Coord(
                in_coord=Coordinate(x=3 + i * 5, y=3),
                out_coord=Coordinate(x=4 + i * 5, y=3),
            )
            for i in range(8)
        }
        return QC_sector_map

    def __set_buffer_sector_coords(self):
        return [Coordinate(x=i, y=6) for i in range(1, 43, 1)]

    def __set_yard_sector_map(self) -> Dict[str, In_Out_Coord]:
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        yard_sector_map = dict()
        for i in range(8):
            letter = letters[i]
            yard_sector_map[f"{letter}1"] = In_Out_Coord(
                in_coord=Coordinate(x=2 + i * 5, y=13),
                out_coord=Coordinate(x=3 + i * 5, y=13),
            )
            yard_sector_map[f"{letter}2"] = In_Out_Coord(
                in_coord=Coordinate(x=4 + i * 5, y=13),
                out_coord=Coordinate(x=5 + i * 5, y=13),
            )

        return yard_sector_map

    def get_sector(self, coord: Coordinate) -> Sector:
        col_id, row_id = coord.x, coord.y
        return self.__data[row_id][col_id]

    def get_QC_sector(self, QC_name: str) -> In_Out_Coord:
        return self.__QC_sector_map.get(QC_name, None)

    def get_buffer_sector_coords(self):
        return self.__buffer_sector_coords

    def get_yard_sector(self, yard_name: str) -> In_Out_Coord:
        return self.__yard_sector_map.get(yard_name, None)

    # support tracking HTs' operations
    def is_sector_available(self, coord: Coordinate) -> bool:
        sector = self.get_sector(coord)
        return sector.is_available()

    def add_occupator(self, coord: Coordinate, HT_name: str):
        sector = self.get_sector(coord)
        if sector.is_available():
            sector.add_occupator(HT_name)
        else:
            raise OverflowError(f"The sector {sector} cannot add more occupator.")

    def remove_occupator(self, coord: Coordinate, HT_name: str):
        sector = self.get_sector(coord)
        if sector:
            sector.remove_occupator(HT_name)
        else:
            raise ValueError(f"Sector at coordinate {coord} does not exist.")

    def move_occupator(
        self, from_coord: Coordinate, to_coord: Coordinate, HT_name: str
    ):
        from_sector = self.get_sector(from_coord)
        to_sector = self.get_sector(to_coord)

        # check move is valid
        from_coord_has_HT = from_sector.has_occupator(HT_name)
        to_coord_not_have_HT = not to_sector.has_occupator(HT_name)
        is_movable = (
            to_sector.get_coordinate() in from_sector.get_movable_to_coordinates()
        )

        if from_coord_has_HT and to_coord_not_have_HT and is_movable:
            from_sector.remove_occupator(HT_name)
            to_sector.add_occupator(HT_name)
        else:
            raise ValueError(
                f"""Invalid occupator move: from_coord_has_HT={from_coord_has_HT}, 
                to_coord_not_have_HT={to_coord_not_have_HT}, is_movable={is_movable}
                """
            )


class SectorMapSnapshot:
    def __init__(self, sector_map: SectorMap):
        self.__sector_map = sector_map

    def get_QC_sector(self, QC_name: str) -> In_Out_Coord:
        return self.__sector_map.get_QC_sector(QC_name)

    def get_buffer_sector_coord(self):
        return self.__sector_map.get_buffer_sector_coord()

    def get_yard_sector(self, yard_name: str) -> In_Out_Coord:
        return self.__sector_map.get_yard_sector(yard_name)

    def get_moveable_to_coordinates(self, coord: Coordinate):
        sector = self.__sector_map.get_sector(coord)
        if sector:
            return sector.get_movable_to_coordinates()

    def get_capacity(self, coord: Coordinate):
        sector = self.__sector_map.get_sector(coord)
        if sector:
            return sector.get_capacity()

    def get_occupators(self, coord: Coordinate):
        sector = self.__sector_map.get_sector(coord)
        if sector:
            return sector.get_occupators()

    def is_available(self, coord: Coordinate):
        sector = self.__sector_map.get_sector(coord)
        if sector:
            return sector.is_available()

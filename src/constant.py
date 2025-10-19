from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple, Union


@dataclass(frozen=True)
class Screen:
    SECTOR_SIZE_PX: int = 32
    NUMBER_OF_SECTORS_X: int = 43
    NUMBER_OF_SECTORS_Y: int = 25
    WIDTH_PX: int = SECTOR_SIZE_PX * NUMBER_OF_SECTORS_X
    HEIGHT_PX: int = SECTOR_SIZE_PX * NUMBER_OF_SECTORS_Y


@dataclass(frozen=True)
class Lane:
    COLOR: Union[str, Tuple[int, ...]]
    SECTOR_RANGE_Y: List[int]


@dataclass(frozen=True)
class Arrow:
    COLOR: Union[str, Tuple[int, ...]] = "black"
    MARGIN: int = 5  # distance between Arrow and the sector outline it's closest to
    LENGTH: int = 32 - 2 * MARGIN


@dataclass(frozen=True)
class Terminal_Floor:
    QC_TRAVEL_LANE: Lane = Lane(
        COLOR=(255, 255, 224), SECTOR_RANGE_Y=[4, 5]
    )  # light yellow
    BUFFER_ZONE_LANE: Lane = Lane(
        COLOR=(255, 224, 189), SECTOR_RANGE_Y=[6]
    )  # light beige
    HIGHWAY_LANE: Lane = Lane(
        COLOR=(211, 211, 211),  # light grey
        SECTOR_RANGE_Y=[
            7,
            8,
            9,
            10,
            11,
            12,
        ],
    )  # type: ignore
    START_SECTOR_X: int = 1
    START_SECTOR_Y: int = 4
    NUMBER_OF_SECTORS_X: int = 43
    NUMBER_OF_SECTORS_Y: int = 9
    RIGHT_ARROW_SECTOR_X: List[int] = field(default_factory=lambda: [4, 5, 8, 10, 12])
    RIGHT_ARROW_COLOR: Union[str, Tuple[int, ...]] = (0, 51, 102)  # dark blue
    LEFT_ARROW_SECTOR_X: List[int] = field(default_factory=lambda: [7, 9, 11])
    LEFT_ARROW_COLOR: Union[str, Tuple[int, ...]] = (102, 153, 204)  # light blue


@dataclass(frozen=True)
class QuayCraneFloor:
    START_SECTOR_X: int = 3
    START_SECTOR_Y: int = 2
    NUMBER_OF_SECTORS_BETWEEN_QC: int = 5
    ARROW_COLOR: Union[str, Tuple[int, ...]] = (34, 139, 34)  # forest green
    QC_NAME_SECTOR_COLOR: Union[str, Tuple[int, ...]] = "white"
    IN_SECTOR_COLOR: Union[str, Tuple[int, ...]] = (181, 101, 29)  # light brown
    OUT_SECTOR_COLOR: Union[str, Tuple[int, ...]] = (173, 216, 230)  # light blue
    IMAGE: Path = Path("images/quay_crane.png")
    QC_NAMES: List[str] = field(
        default_factory=lambda: [f"QC{i}" for i in range(1, 9, 1)]
    )


@dataclass(frozen=True)
class YardFloor:
    START_SECTOR_X: int = 2
    START_SECTOR_Y: int = 13
    NUMBER_OF_SECTORS_BETWEEN_YARD_BLOCK: int = 5
    ARROW_COLOR: Union[str, Tuple[int, ...]] = (34, 139, 34)  # forest green
    YARD_NAME_SECTOR_COLOR: Union[str, Tuple[int, ...]] = "white"
    IN_SECTOR_COLOR: Union[str, Tuple[int, ...]] = (181, 101, 29)  # light brown
    OUT_SECTOR_COLOR: Union[str, Tuple[int, ...]] = (173, 216, 230)  # light blue
    IMAGE: Path = Path("images/yard.png")
    YARD_NAMES: List[str] = field(
        default_factory=lambda: [
            "A1",
            "A2",
            "B1",
            "B2",
            "C1",
            "C2",
            "D1",
            "D2",
            "E1",
            "E2",
            "F1",
            "F2",
            "G1",
            "G2",
            "H1",
            "H2",
        ]
    )


@dataclass(frozen=True)
class CoordinateMap:
    X_RANGE: Tuple[int, int] = (0, 42)
    Y_RANGE: Tuple[int, int] = (0, 13)


@dataclass(frozen=True)
class CellInfo:
    HEADER: str
    CELL_X_RANGE: Tuple[int, int]
    CELL_Y_RANGE: Tuple[int, int] = (17 * 32, 25 * 32)
    HEADER_Y_RANGE: Tuple[int, int] = (17 * 32, 19 * 32)
    ATTR_Y_RANGE: Tuple[int, int] = (19 * 32, 25 * 32)


@dataclass(frozen=True)
class JobTracker:
    WIDTH_PX: int = 32 * 43
    HEIGHT_PX: int = 32 * 8
    START_SECTOR_X: int = 0
    START_SECTOR_Y: int = 17
    JOB_INFO: CellInfo = CellInfo(
        HEADER="JOBS",
        CELL_X_RANGE=(0, 7 * 32),
    )
    # QUAY_CRANE_INFO: CellInfo = CellInfo(
    #     HEADER="QUAY CRANE",
    #     CELL_X_RANGE=(7 * 32, 12 * 32),
    # )
    HT_INFO: CellInfo = CellInfo(
        HEADER="HT",
        CELL_X_RANGE=(7 * 32, 12 * 32),  # (12 * 32, 17 * 32),
    )
    # YARD_INFO: CellInfo = CellInfo(HEADER="YARD", CELL_X_RANGE=(17 * 32, 22 * 32))
    HT_LOCATIONS: CellInfo = CellInfo(
        HEADER="HT LOCATIONS",
        CELL_X_RANGE=(12 * 32, 43 * 32),  # (22 * 32, 43 * 32)
    )


@dataclass(frozen=True)
class JobParameter:
    DISCHARGE_JOB_TYPE: str = "DI"
    LOADED_JOB_TYPE: str = "LO"
    YARD_WORK_TIME_REQUIRED: int = 300
    QC_WORK_TIME_REQUIRED: int = 120
    HT_DRIVE_TIME_PER_SECTOR: int = 10
    SYSTEM_TIME_PASSED: int = 10


@dataclass(frozen=True)
class HTFleet:
    IMAGE_GO_LEFT: str = "ht_left"
    IMAGE_GO_RIGHT: str = "ht_right"
    IMAGE_GO_UP: str = "ht_up"
    IMAGE_GO_DOWN: str = "ht_down"
    HT_NAMES: List[str] = field(
        default_factory=lambda: [f"HT_{i:02}" for i in range(1, 81, 1)]
    )
    HT_INIT_COORDINATES: List[Tuple[int, int]] = field(
        default_factory=lambda: [(i, 6) for i in range(2, 42, 1) for _ in range(2)]
    )
    HT_INIT_PIXEL_LOCATION: List[Any] = field(
        default_factory=lambda: [
            ((i + 5 / 2) * 32, (6 + 1 / 2) * 32) for i in range(40) for _ in range(2)
        ]
    )


@dataclass(frozen=True)
class UIConstant:
    SCREEN: Screen = Screen()
    ARROW: Arrow = Arrow()

    COORDINATE_MAP: CoordinateMap = CoordinateMap()
    TERMINAL_FLOOR: Terminal_Floor = Terminal_Floor()
    QUAY_CRANE_FLOOR: QuayCraneFloor = QuayCraneFloor()
    HT_FLEET: HTFleet = HTFleet()
    YARD_FLOOR: YardFloor = YardFloor()
    JOB_TRACKER: JobTracker = JobTracker()
    JOB_PARAMETER: JobParameter = JobParameter()

    PLANNING_INTERVAL: int = 60  # one minute
    DEADLOCK_THRESHOLD: int = 3600  # one hour


CONSTANT = UIConstant()

from typing import Dict

from pgzero.actor import Actor

from src.constant import CONSTANT
from src.floor import Coordinate


class HTActor:
    """Actor represents for one HT object on screen."""

    def __init__(self, center_x: int, center_y: int):
        self.actor = Actor(
            CONSTANT.HT_FLEET.IMAGE_GO_LEFT,
            (
                center_x,
                center_y,
            ),
        )

    def update(self, next_x: int, next_y: int):
        # update image if there is new change in location
        if (self.actor.x != next_x) or (self.actor.y != next_y):
            if next_x > self.actor.x:
                self.actor.image = CONSTANT.HT_FLEET.IMAGE_GO_RIGHT
            elif next_x < self.actor.x:
                self.actor.image = CONSTANT.HT_FLEET.IMAGE_GO_LEFT
            elif next_y > self.actor.y:
                self.actor.image = CONSTANT.HT_FLEET.IMAGE_GO_DOWN
            else:
                self.actor.image = CONSTANT.HT_FLEET.IMAGE_GO_UP

            # always face left if in buffer
            if next_y == 208:
                self.actor.image = CONSTANT.HT_FLEET.IMAGE_GO_LEFT

        # update coordinate
        self.actor.x, self.actor.y = next_x, next_y

    def draw(self):
        self.actor.draw()

    def __str__(self):
        return f"HTActor({self.actor.x}, {self.actor.y})"


class HTActorFleet:
    """
    Represents a fleet of HT actors, each initialized at specific pixel locations.

    Attributes
    ----------
    HT_actors : dict[str, HTActor]
        A dictionary mapping HT names (e.g., "HT_01", "HT_02", ...) to their corresponding HTActor instances,
        each initialized at a predefined pixel location.
    """

    def __init__(self):
        HT_names = [f"HT_{i:02}" for i in range(1, 81, 1)]
        HT_locations = [loc for loc in CONSTANT.HT_FLEET.HT_INIT_PIXEL_LOCATION]
        self.HT_actors = {
            HT_name: HTActor(center_x=location[0], center_y=location[1])
            for HT_name, location in zip(HT_names, HT_locations)
        }

    def update(self, HT_name_coords: Dict[str, Coordinate]):
        for HT_name, HT_actor in self.HT_actors.items():
            new_coord = HT_name_coords.get(HT_name)
            HT_actor.update(next_x=new_coord.x, next_y=new_coord.y)

    def draw(self):
        for HT_name, HT_actor in self.HT_actors.items():
            HT_actor.draw()

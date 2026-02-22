from dataclasses import dataclass


@dataclass
class Vector2:
    x: float
    y: float

    @staticmethod
    def zero() -> "Vector2":
        return Vector2(0, 0)

@dataclass
class Vector3:
    x: float
    y: float
    z: float

    @staticmethod
    def zero() -> "Vector3":
        return Vector3(0, 0, 0)



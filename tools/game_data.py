import json
from typing import Any
from dataclasses import asdict, dataclass, field
from .utility import Vector3

class ModelId(int):
    pass

class TextureId(int):
    pass

class EntityId(int):
    pass

class LevelId(int):
    pass

class InstanceId(int):
    pass

@dataclass
class _Data:
    models: dict[ModelId, Model] = field(default_factory=dict)
    textures: dict[TextureId, Texture] = field(default_factory=dict)
    entities: dict[EntityId, Entity] = field(default_factory=dict)
    levels: dict[LevelId, Level] = field(default_factory=dict)

_DATA = _Data()

@dataclass
class Model:
    id: ModelId = field(init=False)
    filename: str

    def __post_init__(self):
        self.id = ModelId(len(_DATA.models))
        _DATA.models[self.id] = self

@dataclass
class Texture:
    id: TextureId = field(init=False)
    filename: str

    def __post_init__(self):
        self.id = TextureId(len(_DATA.textures))
        _DATA.textures[self.id] = self

@dataclass
class Entity:
    id: EntityId = field(init=False)
    name: str
    model: ModelId | None
    texture: TextureId | None

    def __post_init__(self):
        if self.model not in _DATA.models:
            raise Exception(f"Attempted to add entity with non-existent model: {self.model}")
        if self.texture not in _DATA.textures:
            raise Exception(f"Attempted to add instance with non-existent texture: {self.texture}")
        self.id = EntityId(len(_DATA.entities))
        _DATA.entities[self.id] = self

@dataclass
class Instance:
    id: InstanceId = field(init=False)
    level: LevelId
    entity: EntityId
    position: Vector3 = field(default_factory=Vector3.zero) 
    velocity: Vector3 = field(default_factory=Vector3.zero)
    rotation: float = 0
    scale: float = 1

    def __post_init__(self):
        if self.level not in _DATA.levels:
            raise Exception(f"Attempted to add instance to non-existent level: {self.level}")
        if self.entity not in _DATA.entities:
            raise Exception(f"Attempted to add instance of non-existent entity: {self.entity}")
        if self.entity not in _DATA.levels[self.level].entity_instances:
            _DATA.levels[self.level].entity_instances[self.entity] = {}
        self.id = InstanceId(len(_DATA.levels[self.level].entity_instances[self.entity]))
        _DATA.levels[self.level].entity_instances[self.entity][self.id] = self

@dataclass
class Level:
    id: LevelId = field(init=False)
    name: str
    entity_instances: dict[EntityId, dict[InstanceId, Instance]] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.id = LevelId(len(_DATA.levels))
        _DATA.levels[self.id] = self

def export(filename: str):
    with open(filename, "w") as f:
        json.dump(asdict(_DATA), f, indent=2, sort_keys=True)


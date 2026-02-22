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
    models: dict[ModelId, Any] = field(default_factory=dict)
    textures: dict[TextureId, Any] = field(default_factory=dict)
    entities: dict[EntityId, Any] = field(default_factory=dict)
    levels: dict[LevelId, Any] = field(default_factory=dict)
    instances: dict[InstanceId, Any] = field(default_factory=dict)

_DATA = _Data()

@dataclass
class Model:
    id: ModelId = field(init=False)
    filename: str

    def __post_init__(self):
        self.id = ModelId(len(_DATA.models))
        _DATA.models[self.id] = asdict(self)

@dataclass
class Texture:
    id: TextureId = field(init=False)
    filename: str

    def __post_init__(self):
        self.id = TextureId(len(_DATA.textures))
        _DATA.textures[self.id] = asdict(self)

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
        _DATA.entities[self.id] = asdict(self)

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
        self.id = InstanceId(len(_DATA.instances))
        _DATA.instances[self.id] = asdict(self)

@dataclass
class Level:
    id: LevelId = field(init=False)
    name: str

    def __post_init__(self):
        self.id = LevelId(len(_DATA.levels))
        _DATA.levels[self.id] = asdict(self)

def export(filename: str):
    with open(filename, "w") as f:
        json.dump(asdict(_DATA), f, indent=2, sort_keys=True)


# class Game:
#     def __init__(self, name: str) -> None:
#         self.name = name
#         self.entities = {}
#         self.models = {}
#         self.textures = {}
#         self.levels = {}
#         self.instances = {}
#
#     def level(self,):
#         pass
#
#     def entity(self, 
#         name: str, 
#         model: ModelId | None, 
#         texture: TextureId | None
#     ) -> EntityId:
#         id = EntityId(len(self.entities))
#         entity = locals()
#         if model not in self.models:
#             raise Exception(f"Model {model} does not exist")
#         if texture not in self.textures:
#             raise Exception(f"Texture {texture} does not exist")
#         self.entities[id] = entity
#         return id
#
#     def instance(self, 
#         entity: EntityId, 
#         position: Vector3 | None = None,
#         velocity: Vector3 | None = None,
#         rotation: float = 0, 
#         scale: float = 1
#     ) -> InstanceId:
#         id = InstanceId(len(self.entities))
#         instance = locals()
#         if entity not in self.entities:
#             raise Exception(f"Entity {entity} does not exist")
#         if position is None: position = Vector3(0, 0, 0)
#         if velocity is None: velocity = Vector3(0, 0, 0)
#         self.entities[entity].instances.append(instance)
#         return id
#
#     def model(self, filename: str) -> ModelId:
#         mid = ModelId(len(self.models))
#         m = Model(mid, filename)
#         self.models[mid] = m
#         return m.id
#
#     def texture(self, filename: str) -> TextureId:
#         tid = TextureId(len(self.textures))
#         t = Texture(tid, filename)
#         self.textures[tid] = t
#         return t.id
#
#     def as_dict(self):
#         return {
#             "name": self.name,
#             "entities": [asdict(e) for e in self.entities.values()],
#             "models": [asdict(m) for m in self.models.values()],
#             "textures": [asdict(t) for t in self.textures.values()]
#         }
#
#     def export(self) -> None:
#         with open(f"{self.name}.json", "w") as f:
#             json.dump(self.as_dict(), f, indent=2, sort_keys=True)
#

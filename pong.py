from tools.game_data import Entity, Instance, Level, Model, Texture, Vector3, export

TOP = -1
BOTTOM = 1
LEFT = -1
RIGHT = -1

def main():
    viking_room_model = Model("viking_room.obj")
    viking_room_texture = Texture("viking_room.png")
    room = Entity("Room", viking_room_model.id, viking_room_texture.id)
    main = Level("Main")
    Instance(main.id, room.id, position=Vector3(-0.8, 0, 0))
    Instance(main.id, room.id, position=Vector3(0.8, 0, 0))

    export("Pong.json")

if __name__ == "__main__":
    main()


from tools.game_data import Entity, Instance, Level, Model, Texture, Vector3, export

TOP = -1
BOTTOM = 1
LEFT = -1
RIGHT = 1

def main():
    # viking_room_model = Model("models/viking_room.obj")
    viking_room_texture = Texture("textures/viking_room.png")

    ball_model = Model("models/ball.obj") 
    paddle_model = Model("models/paddle.obj")

    paddle = Entity("Paddle", paddle_model.id, viking_room_texture.id)
    ball = Entity("Ball", ball_model.id, viking_room_texture.id)

    main = Level("Main")

    Instance(main.id, paddle.id, position=Vector3(-0.5, 0, 0))
    Instance(main.id, paddle.id, position=Vector3(0.5, 0, 0))
    Instance(main.id, ball.id, position=Vector3.zero())

    export("Pong.json")

if __name__ == "__main__":
    main()


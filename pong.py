from tools.gamedata import *
from tools.utility import Vector3

TOP = -1
BOTTOM = 1
LEFT = -1
RIGHT = 1

def main():
    frameSize(1200, 800)

    viking_room_texture = texture("textures/viking_room.png")
    ball_model = model("models/ball.obj") 
    paddle_model = model("models/paddle.obj")

    paddle = entity("Paddle", paddle_model.id, viking_room_texture.id)
    ball = entity("Ball", ball_model.id, viking_room_texture.id)

    main = level("Main")

    instance(main.id, paddle.id, position=Vector3(-0.5, 0, 0))
    instance(main.id, paddle.id, position=Vector3(0.5, 0, 0))
    instance(main.id, ball.id, position=Vector3.zero())

    export("Pong.json")

if __name__ == "__main__":
    main()


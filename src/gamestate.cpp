#include "gamestate.h"
#include <cstdlib>
#include <print>

GameState loadGameState() {
    Model paddleModel{
        .id = 0,
        .filename = "models/paddle.obj"
    };

    Model ballModel{
        .id = 1,
        .filename = "models/ball.obj"
    };

    Model barrierModel{
        .id = 2,
        .filename = "models/barrier.obj"
    };

    Instance playerInstance{
        .id = 0,
        .modelId = paddleModel.id,
        .position = glm::vec3(0.8, 0.0, 0.0),
        .hitBox = { 0.1, 0.5 }
    };

    Instance opponentInstance{
        .id = 1,
        .modelId = paddleModel.id,
        .position = glm::vec3(-0.8, 0.0, 0.0),
        .hitBox = { 0.1, 0.5 }
    };

    Instance ballInstance{
        .id = 2,
        .modelId = ballModel.id,
        .position = glm::vec3(0.0),
        .hitBox = { 0.1, 0.1 }
    };

    Instance leftBarrier{
        .id = 3,
        .modelId = barrierModel.id,
        .position = glm::vec3(-1.0, 0.0, 0.0),
        .hitBox = {0.1, 2.0}
    };

    Instance rightBarrier{
        .id = 3,
        .modelId = barrierModel.id,
        .position = glm::vec3(1.0, 0.0, 0.0),
        .hitBox = {0.1, 2.0}
    };

    Instance topBarrier{
        .id = 4,
        .modelId = barrierModel.id,
        .position = glm::vec3(0.0, 1.0, 0.0),
        .rotation = 90.0,
        .hitBox = {2.0, 0.1}
    };

    Instance bottomBarrier{
        .id = 5,
        .modelId = barrierModel.id,
        .position = glm::vec3(0.0, -1.0, 0.0),
        .rotation = 90.0,
        .hitBox = {2.0, 0.1}
    };

    std::unordered_map<uint32_t, Instance> paddleInstances = {
        {playerInstance.id, playerInstance},
        {opponentInstance.id, opponentInstance}
    };

    std::unordered_map<uint32_t, Instance> ballInstances = {
        {ballInstance.id, ballInstance},
    };

    GameState state{
        .frameWidth = 1200,
        .frameHeight = 1200,
        .ballSpeedUp = 1.1,
        .maxBallSpeed = 4,
        .opponentSpeed = 1,
        .player = playerInstance,
        .opponent = opponentInstance,
        .ball = ballInstance,
        .leftBarrier = leftBarrier,
        .rightBarrier = rightBarrier,
        .topBarrier = topBarrier,
        .bottomBarrier = bottomBarrier,
        .models = {
            {paddleModel.id, paddleModel},
            {ballModel.id, ballModel},
            {barrierModel.id, barrierModel},
        },
    };

    return state;
}

static float randFloat() {
    return (float)rand() / (float)RAND_MAX;
}

static float top(Instance& i) {
    return i.position.y + i.hitBox.height / 2.0;
}

static float bottom(Instance& i) {
    return i.position.y - i.hitBox.height / 2.0;
}

static float left(Instance& i) {
    return i.position.x - i.hitBox.width / 2.0;
}

static float right(Instance& i) {
    return i.position.x + i.hitBox.width / 2.0;
}

static bool hit(Instance& a, Instance& b) {
    bool xIntersect = right(a) >= left(b) && left(a) <= right(b);
    bool yIntersect = top(a) >= bottom(b) && bottom(a) <= top(b);

    return xIntersect && yIntersect;
}

static glm::vec3 hitDirection(Instance& paddle, Instance& ball, bool left) {
    float multiplier = 2.0;
    float x;
    if (left) {
        x = -1.0;
    } else {
        x = 1.0;
    }
    float diff = (ball.position.y - paddle.position.y) * multiplier;
    return glm::normalize(glm::vec3(x, diff, 0.0));
}

void updateGameState(GameState& g, InputState& inputState, float deltaTime) {
    // std::print("\rMouse Position: {:.2f}, {:.2f}", inputState.mousePos.x, inputState.mousePos.y);
    // std::print("\rFrame Size: {}, {}", gameState.frameWidth, gameState.frameHeight);
    std::print("\rBall speed: {:.2f}", glm::length(g.ball.velocity));

    glm::vec2 playerPos{
        inputState.mousePos.x / g.frameWidth * 2.0 - 1.0,
        -(inputState.mousePos.y / g.frameHeight * 2.0 - 1.0),
    }; 

    g.player.position.y = playerPos.y;
    g.ball.position += g.ball.velocity * deltaTime;
    g.opponent.position += g.opponent.velocity * deltaTime;

    // Move opponent
    if (g.opponent.position.y > g.ball.position.y && g.opponent.velocity.y >= 0) {
        g.opponent.velocity.y = -g.opponentSpeed;
    } 

    if (g.opponent.position.y < g.ball.position.y && g.opponent.velocity.y <= 0) {
        g.opponent.velocity.y = g.opponentSpeed;
    }

    // Random start direction for ball
    if (g.ball.velocity == glm::vec3(0.0)) {
        g.ball.velocity = glm::normalize(glm::vec3(randFloat(), randFloat(), 0.0f));
    }

    // Player hit. Player is on the right, only reverse ball direction if ball is moving right
    if (hit(g.player, g.ball) && g.ball.velocity.x > 0) {
        float speed = glm::length(g.ball.velocity);
        glm::vec3 direction = hitDirection(g.player, g.ball, true);
        if (speed >= g.maxBallSpeed) {
            g.ball.velocity = direction * speed;
        } else {
            g.ball.velocity = direction * speed * g.ballSpeedUp;
        }
    }

    // Opponent hit
    if (hit(g.opponent, g.ball) && g.ball.velocity.x < 0) {
        float speed = glm::length(g.ball.velocity);
        glm::vec3 direction = hitDirection(g.player, g.ball, false);
        if (speed >= g.maxBallSpeed) {
            g.ball.velocity = direction * speed;
        } else {
            g.ball.velocity = direction * speed * g.ballSpeedUp;
        }
    }

    // Top barrier
    if (hit(g.topBarrier, g.ball) && g.ball.velocity.y > 0) {
        g.ball.velocity.y = -g.ball.velocity.y;
    }

    // Bottom barrier
    if (hit(g.bottomBarrier, g.ball) && g.ball.velocity.y < 0) {
        g.ball.velocity.y = -g.ball.velocity.y;
    }

    // Right barrier (Opponent goal)
    if (hit(g.rightBarrier, g.ball) && g.ball.velocity.x > 0) {
        g.ball.position = glm::vec3(0.0);
        g.ball.velocity = glm::normalize(glm::vec3(randFloat(), randFloat(), 0.0f));
    }

    // Left barrier (Player goal)
    if (hit(g.leftBarrier, g.ball) && g.ball.velocity.x < 0) {
        g.ball.position = glm::vec3(0.0);
        g.ball.velocity = glm::normalize(glm::vec3(randFloat(), randFloat(), 0.0f));
    }
}


#include "gamestate.h"
#include <cstdlib>
#include <print>
#include <filesystem>
#include <unordered_map>

GameState loadGameState() {
    GameState g{
        .begun = false,
        .frameWidth = 800,
        .frameHeight = 800,
        .ballSpeedUp = 1.1,
        .maxBallSpeed = 4,
        .opponentSpeed = 1
    };

    uint32_t paddleModelId = g.addModel("models/paddle.obj");
    uint32_t ballModelId = g.addModel("models/ball.obj");
    uint32_t barrierModelId = g.addModel("models/barrier.obj");

    // Load font
    for (const auto& entry : std::filesystem::directory_iterator("models/font")) {
        std::string path = entry.path().string();
        std::string stem = entry.path().stem().string();
        char letter = stem[0];
        uint32_t modelId = g.addModel(path);
        g.fontCharModels[letter] = modelId;
    }

    g.message = {
        .text = "CLICK",
        .position = glm::vec3(0),
        .spacing = 0.11,
        .scale = 0.01,
    };

    g.player = {
        .modelId = paddleModelId,
        .position = glm::vec3(0.8, 0.0, 0.0),
        .hitBox = { 0.1, 0.5 },
    };

    g.opponent = {
        .modelId = paddleModelId,
        .position = glm::vec3(-0.8, 0.0, 0.0),
        .hitBox = { 0.1, 0.5 }
    };
    
    g.ball = {
        .modelId = ballModelId,
        .position = glm::vec3(0.0),
        .hitBox = { 0.1, 0.1 },
        .hidden = true
    };

    g.leftBarrier = {
        .modelId = barrierModelId,
        .position = glm::vec3(-1.0, 0.0, 0.0),
        .hitBox = {0.1, 2.0}
    };

    g.rightBarrier = {
        .modelId = barrierModelId,
        .position = glm::vec3(1.0, 0.0, 0.0),
        .hitBox = {0.1, 2.0}
    };

    g.topBarrier = {
        .modelId = barrierModelId,
        .position = glm::vec3(0.0, 1.0, 0.0),
        .rotation = 90.0,
        .hitBox = {2.0, 0.1}
    };

    g.bottomBarrier = {
        .modelId = barrierModelId,
        .position = glm::vec3(0.0, -1.0, 0.0),
        .rotation = 90.0,
        .hitBox = {2.0, 0.1}
    };

    return g;
}

std::vector<Instance> GameState::getInstances() {
    std::vector<Instance> instances = {
        player, opponent, ball, 
        leftBarrier, rightBarrier, topBarrier, bottomBarrier,
    };

    int i = 0;
    for (char c : message.text) {
        glm::vec3 charPosition = glm::vec3(message.position.x + i*message.spacing, message.position.y, message.position.z);
        Instance inst{
            .modelId = fontCharModels[c],
            .position = charPosition,
            .rotation = 0,
            .scale = message.scale,
            .hidden = message.hidden
        };
        instances.push_back(inst);
        i++;
    }

    std::erase_if(instances, [](Instance& i) {
        return i.hidden;
    });

    return instances;
}

uint32_t GameState::addModel(const std::string& filename) {
    uint32_t id = models.size();
    Model m{
        .id = id,
        .filename = filename
    };
    models.push_back(m);
    return m.id;
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
    // std::print("\rBall speed: {:.2f}", glm::length(g.ball.velocity));

    if (!g.begun && inputState.leftMousePressed) {
        std::println("Begin");
        g.begun = true;
        g.ball.hidden = false;
        g.message.hidden = true;
    }

    glm::vec2 mousePos{
        inputState.mousePos.x / g.frameWidth * 2.0 - 1.0,
        -(inputState.mousePos.y / g.frameHeight * 2.0 - 1.0),
    }; 

    g.player.position.y = mousePos.y;

    if (g.begun) {
        g.ball.position += g.ball.velocity * deltaTime;
        g.opponent.position += g.opponent.velocity * deltaTime;
    }

    // Move opponent
    if (g.opponent.position.y > g.ball.position.y && g.opponent.velocity.y >= 0) {
        g.opponent.velocity.y = -g.opponentSpeed;
    } 

    if (g.opponent.position.y < g.ball.position.y && g.opponent.velocity.y <= 0) {
        g.opponent.velocity.y = g.opponentSpeed;
    }

    // Random start direction for ball
    if (g.ball.velocity == glm::vec3(0.0)) {
        // g.ball.velocity = glm::normalize(glm::vec3(randFloat(), randFloat(), 0.0f));
        g.ball.velocity = glm::vec3(-1.0, 0.0, 0.0);
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
        g.ball.velocity = glm::vec3(-1.0, 0.0, 0.0);
    }

    // Left barrier (Player goal)
    if (hit(g.leftBarrier, g.ball) && g.ball.velocity.x < 0) {
        g.ball.position = glm::vec3(0.0);
        g.ball.velocity = glm::vec3(-1.0, 0.0, 0.0);
    }
}


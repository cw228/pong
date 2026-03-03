#pragma once

#include <glm/glm.hpp>
#include <string>
#include <unordered_map>

#include "inputstate.h"

struct HitBox {
    float width;
    float height;
};

struct Instance {
    uint32_t id;
    uint32_t modelId;
    glm::vec3 position;
    glm::vec3 velocity;
    float rotation = 0.0;
    float scale = 1.0;
    HitBox hitBox;
};

struct Model {
    uint32_t id;
    std::string filename;
};

struct Texture {
    uint32_t id;
    std::string filename;
};

struct GameState {
    uint32_t frameWidth;
    uint32_t frameHeight;
    float ballSpeedUp;
    float maxBallSpeed;
    float opponentSpeed;

    Instance player;
    Instance opponent;
    Instance ball;
    Instance leftBarrier;
    Instance rightBarrier;
    Instance topBarrier;
    Instance bottomBarrier;

    std::unordered_map<uint32_t, Model> models;

    std::vector<Instance> getInstances() {
        return {
            player, opponent, ball, 
            leftBarrier, rightBarrier, topBarrier, bottomBarrier
        };
    };
};

GameState loadGameState();
void updateGameState(GameState& gameState, InputState& inputState, float deltaTime);


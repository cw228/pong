#pragma once

#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <print>

#include "inputstate.h"

struct HitBox {
    float width;
    float height;
};

struct Instance {
    uint32_t modelId;
    glm::vec3 position;
    glm::vec3 velocity;
    float rotation = 0.0;
    float scale = 1.0;
    HitBox hitBox;
    bool hidden = false;
};

struct Model {
    uint32_t id;
    std::string filename;
};

struct Texture {
    uint32_t id;
    std::string filename;
};

struct Text {
    std::string text;
    glm::vec3 position;
    float spacing;
    float scale;
    bool hidden = false;
};

struct GameState {
    bool begun;
    uint32_t frameWidth;
    uint32_t frameHeight;
    float ballSpeedUp;
    float maxBallSpeed;
    float opponentSpeed;
    std::unordered_map<char, uint32_t> fontCharModels;

    Instance player;
    Instance opponent;
    Instance ball;
    Instance leftBarrier;
    Instance rightBarrier;
    Instance topBarrier;
    Instance bottomBarrier;

    Text message;

    std::vector<Model> models;

    std::vector<Instance> getInstances();
    uint32_t addModel(const std::string& filename);
};

GameState loadGameState();
void updateGameState(GameState& gameState, InputState& inputState, float deltaTime);


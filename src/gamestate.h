#pragma once

#include <glm/glm.hpp>
#include <string>
#include <unordered_map>

#include "inputstate.h"

struct Entity {
    uint32_t id;
    uint32_t modelId;
    std::string name;
    uint32_t textureId;
};

struct Instance {
    uint32_t id;
    glm::vec3 position;
    glm::vec3 velocity;
    float rotation = 0.0;
    float scale = 1.0;
};

struct Level {
    uint32_t id;
    std::string name;
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, Instance>> entityInstances;
    uint32_t playerEntityId;
    uint32_t playerInstanceId;
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
    std::unordered_map<uint32_t, Entity> entities;
    std::unordered_map<uint32_t, Level> levels;
    std::unordered_map<uint32_t, Model> models;
    std::unordered_map<uint32_t, Texture> textures;
};

GameState loadGameState();
void updateGameState(GameState& gameState, InputState& inputState, float deltaTime);


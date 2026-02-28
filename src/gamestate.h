#pragma once

#include <glm/glm.hpp>
#include <string>
#include <unordered_map>

#include "inputstate.h"

struct Entity {
    int id;
    int model;
    std::string name;
    int texture;
};

struct Instance {
    int id;
    glm::vec3 position;
    glm::vec3 velocity;
    float rotation = 0.0;
    float scale = 1.0;
};

struct Level {
    int id;
    std::string name;
    std::unordered_map<int, std::unordered_map<int, Instance>> entityInstances;
    int playerEntityId;
    int playerInstanceId;
};

struct Model {
    int id;
    std::string filename;
};

struct Texture {
    int id;
    std::string filename;
};

struct GameState {
    int frameWidth;
    int frameHeight;
    std::unordered_map<int, Entity> entities;
    std::unordered_map<int, Level> levels;
    std::unordered_map<int, Model> models;
    std::unordered_map<int, Texture> textures;
};

GameState loadGameState();
void updateGameState(GameState& gameState, InputState& inputState, float deltaTime);


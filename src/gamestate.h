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
    int entity;
    int level;
    glm::vec3 position;
    float rotation;
    float scale;
    glm::vec3 velocity;
};

struct Level {
    int id;
    std::string name;
    std::unordered_map<int, std::unordered_map<int, Instance>> entity_instances;
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
    std::unordered_map<int, Entity> entities;
    std::unordered_map<int, Level> levels;
    std::unordered_map<int, Model> models;
    std::unordered_map<int, Texture> textures;
};

GameState loadGameState(const std::string& path);
void updateGameState(GameState& gameState, InputState& inputState, float deltaTime);


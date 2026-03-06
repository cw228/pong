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

struct Text {
    std::unordered_map<char, uint32_t>& fontModels;
    glm::vec3 position;
    float spacing;
    std::string text;
    float scale;

    std::vector<Instance> getInstances() {
        std::vector<Instance> instances;
        int i = 0;
        for (char c : text) {
            glm::vec3 charPosition = glm::vec3(position.x + i*spacing, position.y, position.z);
            Instance inst{
                .modelId = fontModels[c],
                .position = charPosition,
                .rotation = 180,
                .scale = scale
            };
            instances.push_back(inst);
            i++;
        }
        return instances;
    }
};

struct GameState {
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

    std::vector<Instance> getInstances() {
        std::vector<Instance> instances = {
            player, opponent, ball, 
            leftBarrier, rightBarrier, topBarrier, bottomBarrier,
        };
        // int i = 0;
        // for (char c : text) {
        //     glm::vec3 charPosition = glm::vec3(position.x + i*spacing, position.y, position.z);
        //     Instance inst{
        //         .modelId = fontModels[c],
        //         .position = charPosition,
        //         .rotation = 180,
        //         .scale = scale
        //     };
        //     instances.push_back(inst);
        //     i++;
        // }
        //
        return instances;
    };

    uint32_t addModel(const std::string& filename) {
        Model m{
            .id = static_cast<uint32_t>(models.size()),
            .filename = filename
        };
        models.push_back(m);
        return m.id;
    }
};

GameState loadGameState();
void updateGameState(GameState& gameState, InputState& inputState, float deltaTime);


#include "gamestate.h"
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

    Texture paddleTexture{
        .id = 0,
        .filename = "textures/viking_room_texture.png"
    };

    Entity paddleEntity{
        .id = 0,
        .modelId = paddleModel.id,
        .name = "Paddle",
        .textureId = paddleTexture.id
    };
    
    Entity ballEntity{
        .id = 1,
        .modelId = ballModel.id,
        .name = "Ball",
        .textureId = paddleTexture.id
    };

    Instance playerInstance{
        .id = 0,
        .position = glm::vec3(0.8, 0.0, 0.0),
    };

    Instance opponentInstance{
        .id = 1,
        .position = glm::vec3(-0.8, 0.0, 0.0),
    };

    Instance ballInstance{
        .id = 2,
        .position = glm::vec3(0.0)
    };

    std::unordered_map<int, Instance> paddleInstances = {
        {playerInstance.id, playerInstance},
        {opponentInstance.id, opponentInstance}
    };

    std::unordered_map<int, Instance> ballInstances = {
        {ballInstance.id, ballInstance},
    };

    std::unordered_map<int, std::unordered_map<int, Instance>> entityInstances = {
        {paddleEntity.id, paddleInstances},
        {ballEntity.id, ballInstances}
    };

    Level level1{
        .id = 0,
        .name = "Level 1",
        .entityInstances = entityInstances,
    };

    GameState state{
        .frameWidth = 1200,
        .frameHeight = 800,
        .entities = {
            {paddleEntity.id, paddleEntity},
            {ballEntity.id, ballEntity}
        },
        .levels = {
            {level1.id, level1},
        },
        .models = {
            {paddleModel.id, paddleModel},
            {ballModel.id, ballModel}
        },
        .textures = {
            {paddleTexture.id, paddleTexture}
        }
    };

    return state;
}

void updateGameState(GameState& gameState, InputState& inputState, float deltaTime) {
    // std::print("\rMouse Position: {:.2f}, {:.2f}", inputState.mousePos.x, inputState.mousePos.y);
    // std::print("\rFrame Size: {}, {}", gameState.frameWidth, gameState.frameHeight);
    glm::vec2 mousePos{
        inputState.mousePos.x / gameState.frameWidth * 2.0 - 1.0,
        -(inputState.mousePos.y / gameState.frameHeight * 2.0 - 1.0),
    }; 
    std::print("\rMouse Position NDC: {}, {}", mousePos.x, mousePos.y);
    Level& level = gameState.levels[0];

    // gameState.entities[]
}

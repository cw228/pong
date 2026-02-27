#include "gamestate.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <print>

using json = nlohmann::json;

static const std::vector<Model> models = {
    { .id = 0, .filename = "models/paddle.obj" },
    { .id = 1, .filename = "models/ball.obj" },
};

static const std::vector<Model> textures = {
    { .id = 0, .filename = "textures/viking_room_texture.png" }
};

static const std::vector<Entity> entities = {
    { .id = 0, .model = 0, .name = "paddle", .texture = 0 },
    { .id = 1, .model = 1, .name = "ball",   .texture = 0 },
};
 
static const std::vector<Level> levels = {
    { 
        .id = 0, 
        .name = "Main", 
        .entity_instances {

        } 
    },
};

static glm::vec3 parseVec3(const json& j) {
    return { j.at("x").get<float>(), j.at("y").get<float>(), j.at("z").get<float>() };
}

GameState loadDefaultGameState() {
    GameState state;
    state.frameWidth = 1200;
    state.frameHeight = 800;
    
    return state;
}

GameState loadGameState(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Failed to open game state file: " + path);

    json j = json::parse(file);
    GameState state;

    state.frameWidth = j.at("frameWidth");
    state.frameHeight = j.at("frameHeight");

    for (auto& [key, val] : j.at("entities").items()) {
        Entity e;
        e.id      = val.at("id").get<int>();
        e.model   = val.at("model").get<int>();
        e.name    = val.at("name").get<std::string>();
        e.texture = val.at("texture").get<int>();
        state.entities[e.id] = e;
    }

    for (auto& [key, val] : j.at("levels").items()) {
        Level lvl;
        lvl.id   = val.at("id").get<int>();
        lvl.name = val.at("name").get<std::string>();

        for (auto& [entity_id, instances] : val.at("entity_instances").items()) {
            for (auto& [instance_id, instance] : instances.items()) {
                Instance inst;
                inst.id       = instance.at("id").get<int>();
                inst.entity   = instance.at("entity").get<int>();
                inst.level    = instance.at("level").get<int>();
                inst.position = parseVec3(instance.at("position"));
                inst.rotation = instance.at("rotation").get<float>();
                inst.scale    = instance.at("scale").get<float>();
                inst.velocity = parseVec3(instance.at("velocity"));

                lvl.entity_instances[inst.entity][inst.id] = inst;
            }
        }

        state.levels[lvl.id] = lvl;
    }

    for (auto& [key, val] : j.at("models").items()) {
        Model m;
        m.id       = val.at("id").get<int>();
        m.filename = val.at("filename").get<std::string>();
        state.models[m.id] = m;
    }

    for (auto& [key, val] : j.at("textures").items()) {
        Texture t;
        t.id       = val.at("id").get<int>();
        t.filename = val.at("filename").get<std::string>();
        state.textures[t.id] = t;
    }

    return state;
}

void updateGameState(GameState& gameState, InputState& inputState, float deltaTime) {
    // std::print("\rMouse Position: {:.2f}, {:.2f}", inputState.mousePos.x, inputState.mousePos.y);
    // std::print("\rFrame Size: {}, {}", gameState.frameWidth, gameState.frameHeight);
    glm::vec2 playerPos{
        inputState.mousePos.x / gameState.frameWidth * 2.0 - 1.0,
        inputState.mousePos.y / gameState.frameHeight * 2.0 - 1.0,
    }; 
    std::print("\rPlayer Position: {}, {}", playerPos.x, playerPos.y);
    gameState.entities[]
}

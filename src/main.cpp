#include "renderer.h"
#include "window.h"
#include "gamestate.h"

#include <iostream>
#include <chrono>

int WIDTH = 800;
int HEIGHT = 800;

int main() {
    // TODO: set width and height in gameState
    GameState gameState = loadGameState("Pong.json");
    Window window{gameState.frameWidth, gameState.frameHeight};
    InputState inputState;

    try {
        Renderer renderer(window, gameState);

        auto lastTime = std::chrono::high_resolution_clock::now();

        while (!glfwWindowShouldClose(window)) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
            lastTime = currentTime;

            glfwPollEvents();
            updateInputState(inputState, window);
            updateGameState(gameState, inputState, deltaTime);
            renderer.drawFrame(gameState);
        }

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


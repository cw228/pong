#include "renderer.h"
#include "window.h"
#include "gamestate.h"

#include <iostream>
#include <chrono>

int main() {
    Window window{800, 600};
    InputState inputState;

    try {
        GameState gameState = loadGameState("Pong.json");
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


#include "renderer.h"
#include "window.h"
#include "gamestate.h"

#include <iostream>
#include <chrono>
#include <print>

int WIDTH = 800;
int HEIGHT = 800;

int main() {
    GameState gameState = loadGameState();
    Window window{gameState.frameWidth, gameState.frameHeight};
    InputState inputState;

    try {
        Renderer renderer(window, gameState);

        auto lastTime = std::chrono::high_resolution_clock::now();
        int frameCount = 0;
        float fpsTimer = 0.0f;

        while (!glfwWindowShouldClose(window)) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
            lastTime = currentTime;

            fpsTimer += deltaTime;
            frameCount++;
            if (fpsTimer >= 1.0f) {
                // std::print("\rFPS: {}", frameCount);
                frameCount = 0;
                fpsTimer -= 1.0f;
            }

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


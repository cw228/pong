#include "renderer.h"
#include "window.h"
#include "gamestate.h"

#include <iostream>

void updateGameState(GameState& gameState) {
}

void updateRenderState(RenderState& renderState, GameState& gameState) {
}

int main() {
    Window window(800, 600);

    try {
        GameState gameState = loadGameState("Pong.json");
        Renderer renderer(window, gameState);
        RenderState renderState;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            updateGameState(gameState);
            updateRenderState(renderState, gameState);
            renderer.drawFrame(renderState);
        }

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


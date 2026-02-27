#include "inputstate.h"
#include <print>

void updateInputState(InputState& inputState, Window& window) {
    for (int key = 0; key < GLFW_KEY_LAST; ++key) {
        inputState.keys[key] = glfwGetKey(window, key) == GLFW_PRESS;
    }
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    inputState.mousePos = glm::vec2(x, y);
    // std::print("\rMouse Position: {:.2f}, {:.2f}", inputState.mousePos.x, inputState.mousePos.y);
    std::fflush(stdout);
}


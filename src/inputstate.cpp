#include "inputstate.h"

void updateInputState(InputState& inputState, Window& window) {
    for (int key = 0; key < GLFW_KEY_LAST; ++key) {
        inputState.keys[key] = glfwGetKey(window, key) == GLFW_PRESS;
    }
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    inputState.mousePos = glm::vec2(x, y);
}


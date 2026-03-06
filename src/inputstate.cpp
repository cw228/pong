#include "inputstate.h"
#include <GLFW/glfw3.h>
#include <print>

void updateInputState(InputState& inputState, Window& window) {
    for (int key = 0; key < GLFW_KEY_LAST; ++key) {
        inputState.keys[key] = glfwGetKey(window, key) == GLFW_PRESS;
    }
    int leftMouseState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
    inputState.leftMousePressed = leftMouseState == GLFW_PRESS;
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    inputState.mousePos = glm::vec2(x, y);
}


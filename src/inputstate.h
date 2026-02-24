#include "window.h"

struct InputState {
    glm::vec2 mousePos;
    bool keys[GLFW_KEY_LAST];
};

void updateInputState(InputState& inputState, Window& window);


#include "renderer.h"
#include "window.h"

#include <iostream>

int main() {
    Window window(800, 600);

    try {
        Renderer renderer(window);

        while (!window.closed()) {
            renderer.drawFrame();
        }

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

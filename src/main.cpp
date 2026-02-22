#include "renderer.h"

#include <iostream>

int main() {
    Renderer renderer;

    try {
        renderer.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


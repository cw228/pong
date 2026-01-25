import vulkan_hpp;
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>

class Pong {
    public:
        void run() {

        }

    private:
        void initVulkan() {

        }

        void mainLoop() {

        }

        void cleanup() {

        }

};

int main() {
    Pong pong;

    try {
        pong.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

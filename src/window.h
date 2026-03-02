#pragma once

#include <GLFW/glfw3.h>

struct Window {
    GLFWwindow* handle;

    Window(uint32_t width, uint32_t height) {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHintString(GLFW_WAYLAND_APP_ID, "game");
        handle = glfwCreateWindow(width, height, "Pong", nullptr, nullptr);
    }

    operator GLFWwindow*() const { return handle; }

    ~Window() {
        glfwDestroyWindow(handle);
        glfwTerminate();
    }
};


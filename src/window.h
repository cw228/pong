#pragma once

#include <GLFW/glfw3.h>

struct Window {
    GLFWwindow* handle;

    Window(int width, int height) {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHintString(GLFW_WAYLAND_APP_ID, "game");
        handle = glfwCreateWindow(width, height, "Pong", nullptr, nullptr);
    }

    operator GLFWwindow*() const { return handle; }

    bool closed() {
        glfwPollEvents();
        return glfwWindowShouldClose(handle);
    }

    ~Window() {
        glfwDestroyWindow(handle);
        glfwTerminate();
    }
};


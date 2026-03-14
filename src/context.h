#pragma once

#include "vulkan/vulkan_raii.hpp"
#include "window.h"

struct QueueFamilies {
    uint32_t graphics;
    uint32_t presentation;
};

struct Queues {
    vk::raii::Queue graphics = nullptr;
    vk::raii::Queue presentation = nullptr;
};

struct VulkanContext {
    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    Queues queues;
    QueueFamilies queueFamilies;

    VulkanContext(Window& window);
};


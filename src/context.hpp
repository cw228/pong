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

    // VulkanContext(Window& window) {
    //     instance = createInstance(context);
    //     if (enableValidationLayers) {
    //         debugMessenger = createDebugMessenger(instance, debugCallback);
    //     }
    //     surface = createSurface(instance, window);
    //     physicalDevice = choosePhysicalDevice(instance);
    //     device = createLogicalDevice(physicalDevice, queueFamilies.graphics);
    //     QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice, surface);
    //     QueueFamilies families{ .graphics = queueFamilyIndices.graphics, .presentation = queueFamilyIndices.presentation };// oof
    //     queues.graphics = getQueue(device, families.graphics);
    //     queues.presentation = getQueue(device, families.presentation);
    //     queueFamilies = families;
    // }
};


#pragma once

#include "vulkan/vulkan_raii.hpp"

#include "context.h"

struct RenderTargets {
    vk::raii::SwapchainKHR swapchain = nullptr;
    std::vector<vk::Image> images;
    std::vector<vk::raii::ImageView> imageViews;
    vk::SurfaceFormatKHR surfaceFormat;
    vk::PresentModeKHR presentMode;
    vk::Extent2D extent;

    vk::raii::Image colorImage = nullptr;
    vk::raii::ImageView colorImageView = nullptr;
    vk::raii::DeviceMemory colorImageMemory = nullptr;

    vk::raii::Image depthImage = nullptr;
    vk::raii::ImageView depthImageView = nullptr;
    vk::raii::DeviceMemory depthImageMemory = nullptr;

    std::vector<vk::raii::Semaphore> renderCompleteSemaphores;

    RenderTargets(const VulkanContext& vContext, vk::raii::SwapchainKHR oldSwapchain = nullptr);
};

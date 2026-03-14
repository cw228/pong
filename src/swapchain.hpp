#pragma once

#include "context.hpp"
#include "window.h"
#include "vulkan/vulkan_raii.hpp"

struct Swapchain {
    vk::raii::SwapchainKHR swapchain = nullptr;
    std::vector<vk::raii::ImageView> imageViews;
    vk::SurfaceFormatKHR imageFormat;
    vk::PresentModeKHR presentMode;
    vk::Extent2D extent;
    
    Swapchain(const VulkanContext& cxt, const Window& window, vk::raii::SwapchainKHR oldSwapchain = nullptr) {
        // Choose surface format
        // Choose present mode
        // Choose extent
    }
};

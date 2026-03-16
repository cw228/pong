#include <limits>
#include <algorithm>

#include "vulkan/vulkan_raii.hpp"

#include "context.h"
#include "window.h"

vk::raii::ImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspectMask, uint32_t mipLevels) {
    vk::ImageViewCreateInfo createInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = aspectMask,
            .baseMipLevel = 0,
            .levelCount = mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };

    return vk::raii::ImageView(vContext.device, createInfo);
}

void createSwapchainImageViews() {
    swapchainImageViews.clear();

    for (vk::Image image : swapchainImages) {
        swapchainImageViews.push_back(createImageView(image, swapchainImageFormat, vk::ImageAspectFlagBits::eColor, 1));
    }
}

struct Swapchain {
    vk::raii::SwapchainKHR swapchain = nullptr;
    std::vector<vk::Image> images;
    std::vector<vk::raii::ImageView> imageViews;
    vk::SurfaceFormatKHR surfaceFormat;
    vk::PresentModeKHR presentMode;
    vk::Extent2D extent;

    Swapchain(const VulkanContext& vContext, const Window& window, vk::raii::SwapchainKHR oldSwapchain = nullptr) {
        std::vector<vk::SurfaceFormatKHR> formats = vContext.physicalDevice.getSurfaceFormatsKHR(vContext.surface);
        std::vector<vk::PresentModeKHR> modes = vContext.physicalDevice.getSurfacePresentModesKHR(vContext.surface);
        vk::SurfaceCapabilitiesKHR capabilities = vContext.physicalDevice.getSurfaceCapabilitiesKHR(vContext.surface);        

        // Choose surface format
        surfaceFormat = formats[0];

        for (const vk::SurfaceFormatKHR& format : formats) {
            if (format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                surfaceFormat = format;
                break;
            }
        }

        // Choose present mode
        presentMode = vk::PresentModeKHR::eFifo;

        for (const vk::PresentModeKHR& mode : modes) {
            if (mode == vk::PresentModeKHR::eMailbox) {
                presentMode = mode;
                break;
            }
        }

        // Choose extent
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            extent = capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            extent = {
                std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
            };
        }

        uint32_t minImageCount = std::max(3u, capabilities.minImageCount);
        if (capabilities.maxImageCount > 0 && minImageCount > capabilities.maxImageCount) {
            minImageCount = capabilities.maxImageCount;
        }

        // Create swapchain
        vk::SwapchainCreateInfoKHR createInfo{
            .flags = vk::SwapchainCreateFlagsKHR(),
            .surface = *vContext.surface,
            .minImageCount = minImageCount,
            .imageFormat = surfaceFormat.format,
            .imageColorSpace = surfaceFormat.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            .preTransform = capabilities.currentTransform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = presentMode,
            .clipped = true,
            .oldSwapchain = nullptr
        };

        if (vContext.queueFamilies.graphics != vContext.queueFamilies.presentation) {
            uint32_t queueFamilies[] = { vContext.queueFamilies.graphics, vContext.queueFamilies.presentation };
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilies;
        } else {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        }

        swapchain = vk::raii::SwapchainKHR(vContext.device, createInfo);

        images = swapchain.getImages();

        // Create image views
        for (vk::Image& image : images) {

        }

        // Create multisample color image
        // Create multisample depth image
        // Create semaphores

        // vk::raii::SwapchainKHR swapchain = createSwapchain(
        //     vContext.device, vContext.surface, swapchainImageFormat, swapchainPresentMode, swapchainExtent, capabilities, vContext.queueFamilies
        // );

    }
};


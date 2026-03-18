#include <limits>
#include <algorithm>

#include "render_targets.h"

RenderTargets::RenderTargets(const VulkanContext& vContext, vk::raii::SwapchainKHR oldSwapchain) {
    std::vector<vk::SurfaceFormatKHR> formats = vContext.physicalDevice.getSurfaceFormatsKHR(vContext.surface);
    std::vector<vk::PresentModeKHR> modes = vContext.physicalDevice.getSurfacePresentModesKHR(vContext.surface);
    vk::SurfaceCapabilitiesKHR capabilities = vContext.physicalDevice.getSurfaceCapabilitiesKHR(vContext.surface);        

    // -- Choose surface format --
    surfaceFormat = formats[0];

    for (const vk::SurfaceFormatKHR& format : formats) {
        if (format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            surfaceFormat = format;
            break;
        }
    }

    // -- Choose present mode --
    presentMode = vk::PresentModeKHR::eFifo;

    for (const vk::PresentModeKHR& mode : modes) {
        if (mode == vk::PresentModeKHR::eMailbox) {
            presentMode = mode;
            break;
        }
    }

    // -- Choose extent --
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        extent = capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(vContext.window, &width, &height);
        extent = {
            std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
        };
    }

    // -- Set minimum image count --
    uint32_t minImageCount = std::max(3u, capabilities.minImageCount);
    if (capabilities.maxImageCount > 0 && minImageCount > capabilities.maxImageCount) {
        minImageCount = capabilities.maxImageCount;
    }

    // -- Create swapchain --
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
            .oldSwapchain = *oldSwapchain
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

    // -- Create swapchain image views --
    for (vk::Image& image : images) {
        vk::ImageViewCreateInfo viewCreateInfo{
            .image = image,
                .viewType = vk::ImageViewType::e2D,
                .format = surfaceFormat.format,
                .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                }
        };

        imageViews.push_back(vk::raii::ImageView(vContext.device, viewCreateInfo));
    }

    // -- Create multisample color resources --
    vk::ImageCreateInfo colorImageInfo{
        .imageType = vk::ImageType::e2D,
            .format = surfaceFormat.format,
            .extent = { extent.width, extent.height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vContext.msaaSamples,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eColorAttachment,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined
    };

    colorImage = vk::raii::Image(vContext.device, colorImageInfo);

    vk::MemoryRequirements colorImageMemRequirements = colorImage.getMemoryRequirements();
    vk::MemoryAllocateInfo colorAllocInfo{
        .allocationSize = colorImageMemRequirements.size,
            .memoryTypeIndex = vContext.findMemoryType(colorImageMemRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
    };

    colorImageMemory = vk::raii::DeviceMemory(vContext.device, colorAllocInfo);
    colorImage.bindMemory(colorImageMemory, 0);

    vk::ImageViewCreateInfo colorImageViewInfo{
        .image = colorImage,
        .viewType = vk::ImageViewType::e2D,
        .format = surfaceFormat.format,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    colorImageView = vk::raii::ImageView(vContext.device, colorImageViewInfo);

    // -- Create multisample depth resources --
    vk::ImageCreateInfo depthImageInfo{
        .imageType = vk::ImageType::e2D,
            .format = vk::Format::eD32Sfloat,
            .extent = { extent.width, extent.height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vContext.msaaSamples,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined
    };

    depthImage = vk::raii::Image(vContext.device, depthImageInfo);

    vk::MemoryRequirements depthImageMemRequirements = depthImage.getMemoryRequirements();
    vk::MemoryAllocateInfo depthAllocInfo{
        .allocationSize = depthImageMemRequirements.size,
            .memoryTypeIndex = vContext.findMemoryType(depthImageMemRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
    };

    depthImageMemory = vk::raii::DeviceMemory(vContext.device, depthAllocInfo);
    depthImage.bindMemory(depthImageMemory, 0);

    vk::ImageViewCreateInfo depthImageViewInfo{
        .image = depthImage,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eD32Sfloat,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eDepth,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    depthImageView = vk::raii::ImageView(vContext.device, depthImageViewInfo);

    // -- Create semaphores --
    for (size_t i = 0; i < images.size(); i++) {
        renderCompleteSemaphores.emplace_back(vContext.device, vk::SemaphoreCreateInfo{});
    }
}


// #include "context.h"
// #include "window.h"
// #include "vulkan/vulkan_raii.hpp"
//
// vk::SurfaceFormatKHR chooseSwapSurfaceFormat(vk::raii::PhysicalDevice& physicalDevice, vk::raii::SurfaceKHR& surface) {
//     std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(surface);
//     for (const vk::SurfaceFormatKHR& format : availableFormats) {
//         if (format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
//             return format;
//         }
//     }
//
//     return availableFormats[0];
// }
//
// vk::PresentModeKHR chooseSwapPresentMode(vk::raii::PhysicalDevice& physicalDevice, vk::raii::SurfaceKHR& surface) {
//     std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface);
//     for (const vk::PresentModeKHR& presentMode : availablePresentModes) {
//         if (presentMode == vk::PresentModeKHR::eMailbox) {
//             return presentMode;
//         }
//     }
//
//     return vk::PresentModeKHR::eFifo;
// }
//
// vk::Extent2D clampedExtent(vk::SurfaceCapabilitiesKHR& capabilities, int& width, int& height) {
//     return {
//         std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
//         std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
//     };
// }
//
// vk::Extent2D chooseSwapExtent(Window& window, vk::SurfaceCapabilitiesKHR surfaceCapabilities) {
//     if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
//         return surfaceCapabilities.currentExtent;
//     }
//     int width, height;
//     glfwGetFramebufferSize(window, &width, &height);
//     return clampedExtent(surfaceCapabilities, width, height);
// }
//
// vk::raii::SwapchainKHR createSwapchain(
//     vk::raii::Device& device,
//     vk::raii::SurfaceKHR& surface,
//     vk::SurfaceFormatKHR format,
//     vk::PresentModeKHR presentMode,
//     vk::Extent2D extent,
//     vk::SurfaceCapabilitiesKHR surfaceCapabilities,
//     QueueFamilies queueFamilyIndices
// ) {
//     auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
//     if (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) {
//         minImageCount = surfaceCapabilities.maxImageCount;
//     }
//
//     vk::SwapchainCreateInfoKHR swapchainCreateInfo{
//         .flags = vk::SwapchainCreateFlagsKHR(),
//         .surface = *surface,
//         .minImageCount = minImageCount,
//         .imageFormat = format.format,
//         .imageColorSpace = format.colorSpace,
//         .imageExtent = extent,
//         .imageArrayLayers = 1,
//         .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
//         .preTransform = surfaceCapabilities.currentTransform,
//         .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
//         .presentMode = presentMode,
//         .clipped = true,
//         .oldSwapchain = nullptr
//     };
//
//     if (queueFamilyIndices.graphics != queueFamilyIndices.presentation) {
//         uint32_t queueFamilyIndicesArray[] = { queueFamilyIndices.graphics, queueFamilyIndices.presentation };
//         swapchainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
//         swapchainCreateInfo.queueFamilyIndexCount = 2;
//         swapchainCreateInfo.pQueueFamilyIndices = queueFamilyIndicesArray;
//     } else {
//         swapchainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
//     }
//
//     return vk::raii::SwapchainKHR(device, swapchainCreateInfo);
// }
//
// void Renderer::createSwapchainImageViews() {
//     swapchainImageViews.clear();
//
//     for (vk::Image image : swapchainImages) {
//         swapchainImageViews.push_back(createImageView(image, swapchainImageFormat, vk::ImageAspectFlagBits::eColor, 1));
//     }
// }

// struct Swapchain {
//     vk::raii::SwapchainKHR swapchain = nullptr;
//     std::vector<vk::raii::ImageView> imageViews;
//     vk::SurfaceFormatKHR imageFormat;
//     vk::PresentModeKHR presentMode;
//     vk::Extent2D extent;
//
//     Swapchain(const VulkanContext& cxt, const Window& window, vk::raii::SwapchainKHR oldSwapchain = nullptr) {
//         // Choose surface format
//         vk::SurfaceFormatKHR swapchainImageFormat = chooseSwapSurfaceFormat(vContext.physicalDevice, vContext.surface);
//
//         // Choose present mode
//         vk::PresentModeKHR swapchainPresentMode = chooseSwapPresentMode(vContext.physicalDevice, vContext.surface);
//
//         // Choose extent
//         vk::SurfaceCapabilitiesKHR surfaceCapabilities = vContext.physicalDevice.getSurfaceCapabilitiesKHR(vContext.surface);
//         vk::Extent2D swapchainExtent = chooseSwapExtent(window, surfaceCapabilities);
//
//
//         vk::raii::SwapchainKHR swapchain = createSwapchain(
//             vContext.device, vContext.surface, swapchainImageFormat, swapchainPresentMode, swapchainExtent, surfaceCapabilities, vContext.queueFamilies
//         );
//
//     }
// };

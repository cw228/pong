#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <print>
#include <fstream>

#ifndef GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_VULKAN
#endif
#include <GLFW/glfw3.h>

#ifndef VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#endif
#include <vulkan/vulkan_raii.hpp>


constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

#ifdef __APPLE__
constexpr bool macOS = true;
#else
constexpr bool macOS = false;
#endif

struct QueueFamilyIndices {
    uint32_t graphicsIndex;
    uint32_t presentationIndex;
};

class Pong {
    public:
        void run() {
            initWindow();
            initVulkan();
            mainLoop();
            cleanup();
        }

    private:
        // mk:members
        GLFWwindow* window;
        vk::raii::Context context;
        vk::raii::Instance instance = nullptr;
        vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
        vk::raii::PhysicalDevice physicalDevice = nullptr;
        vk::raii::SurfaceKHR surface = nullptr;
        vk::raii::Device device = nullptr;
        vk::raii::Queue graphicsQueue = nullptr;
        vk::raii::Queue presentQueue = nullptr;
        vk::PhysicalDeviceFeatures deviceFeatures;
        std::vector<const char*> deviceExtensions = {
            vk::KHRSwapchainExtensionName,
#ifdef __APPLE__
            "VK_KHR_portability_subset",
            vk::KHRSynchronization2ExtensionName,
            vk::KHRDynamicRenderingExtensionName,
#endif
        };
        QueueFamilyIndices queueFamilyIndices;
        vk::raii::SwapchainKHR swapchain = nullptr;
        std::vector<vk::Image> swapchainImages;
        vk::Format swapchainImageFormat = vk::Format::eUndefined;
        vk::Extent2D swapchainExtent;
        std::vector<vk::raii::ImageView> swapchainImageViews;
        vk::raii::PipelineLayout pipelineLayout = nullptr;
        vk::raii::Pipeline graphicsPipeline = nullptr;
        vk::raii::CommandPool commandPool = nullptr;

        std::vector<vk::raii::CommandBuffer> commandBuffers;
        std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
        std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
        std::vector<vk::raii::Fence> drawFences;
        uint32_t frameIndex = 0;
        
        void initWindow() {
            glfwInit();

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            // glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

            window = glfwCreateWindow(WIDTH, HEIGHT, "Pong", nullptr, nullptr);
        }

        void initVulkan() {
            createInstance();
            setupDebugMessenger();
            createSurface();
            pickPhysicalDevice();
            findQueueFamilies();
            createLogicalDevice();
            getQueues();
            createSwapchain();
            createImageViews();
            createGraphicsPipeline();
            createCommandPool();
            createCommandBuffers();
            createSyncObjects();
        }

        void mainLoop() {
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                drawFrame();
            }

            device.waitIdle();
        }

        void drawFrame() {
            vk::Result fenceResult = device.waitForFences(*drawFences[frameIndex], vk::True, UINT64_MAX);

            if (fenceResult != vk::Result::eSuccess) {
                throw std::runtime_error("failed to wait for fence");
            }

            // int width, height;
            // glfwGetFramebufferSize(window, &width, &height);
            // vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
            // vk::Extent2D currentExtent = clampedExtent(surfaceCapabilities, width, height);
            //
            // if (currentExtent.width != swapchainExtent.width || currentExtent.height != swapchainExtent.height) {
            //     std::println("Extent changed {}x{} -> {}x{}", swapchainExtent.width, swapchainExtent.height, width, height);
            //     recreateSwapchain();
            // }

            auto [acquireResult, imageIndex] = swapchain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

            if (acquireResult == vk::Result::eErrorOutOfDateKHR) {
                std::println("swapchain out of date!");
            }

            if (acquireResult == vk::Result::eSuboptimalKHR) {
                std::println("swapchain suboptimal!");
            }

            if (acquireResult != vk::Result::eSuccess) {
                throw std::runtime_error("failed to acquire swapchain image");
            }

            recordCommandBuffer(imageIndex);
            device.resetFences(*drawFences[frameIndex]);

            vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
            const vk::SubmitInfo submitInfo{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
                .pWaitDstStageMask = &waitDestinationStageMask,
                .commandBufferCount = 1,
                .pCommandBuffers = &*commandBuffers[frameIndex],
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
            };

            graphicsQueue.submit(submitInfo, *drawFences[frameIndex]);

            const vk::PresentInfoKHR presentInfo{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
                .swapchainCount = 1,
                .pSwapchains = &*swapchain,
                .pImageIndices = &imageIndex
            };

            vk::Result presentResult = presentQueue.presentKHR(presentInfo);

            frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        void recordCommandBuffer(uint32_t imageIndex) {
            vk::raii::CommandBuffer& commandBuffer = commandBuffers[frameIndex];

            commandBuffer.begin({});

            transitionImageLayout(
                imageIndex,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal,
                {},
                vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput
            );

            vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);

            vk::RenderingAttachmentInfo attachmentInfo = {
                .imageView = swapchainImageViews[imageIndex],
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = clearColor
            };

            vk::RenderingInfo renderingInfo = {
                .renderArea = { .offset = { 0, 0 }, .extent = swapchainExtent },
                .layerCount = 1,
                .colorAttachmentCount = 1,
                .pColorAttachments = &attachmentInfo
            };

            commandBuffer.beginRendering(renderingInfo);
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

            vk::Viewport viewport{
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(swapchainExtent.width),
                .height = static_cast<float>(swapchainExtent.height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f
            };

            vk::Rect2D scissor{
                .offset = vk::Offset2D{ 0, 0 },
                .extent = swapchainExtent
            };

            commandBuffer.setViewport(0, viewport);
            commandBuffer.setScissor(0, scissor);
            commandBuffer.draw(3, 1, 0, 0);
            commandBuffer.endRendering();

            transitionImageLayout(
                imageIndex,
                vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageLayout::ePresentSrcKHR,
                vk::AccessFlagBits2::eColorAttachmentWrite,
                {},
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::PipelineStageFlagBits2::eBottomOfPipe
            );

            commandBuffer.end();
        }

        void cleanup() {
            glfwDestroyWindow(window);
        }

        // Vulkan initialization
        void createInstance() {
            constexpr vk::ApplicationInfo appInfo{ 
                .pApplicationName = "Pong",
                .applicationVersion = VK_MAKE_VERSION( 1, 0, 0 ),
                .pEngineName        = "No Engine",
                .engineVersion      = VK_MAKE_VERSION( 1, 0, 0 ),
                .apiVersion         = vk::ApiVersion14 
            };

            std::vector<const char*> requiredLayers = getRequiredLayers();
            ensureLayersSupported(requiredLayers);

            std::vector<const char*> requiredExtentions = getRequiredExtentions();
            ensureExtensionsSupported(requiredExtentions);

            vk::InstanceCreateInfo createInfo{
                .pApplicationInfo = &appInfo,
                .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
                .ppEnabledLayerNames = requiredLayers.data(),
                .enabledExtensionCount = static_cast<uint32_t>(requiredExtentions.size()),
                .ppEnabledExtensionNames = requiredExtentions.data()
            };

            if (macOS) {
                createInfo.flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
            }

            instance = vk::raii::Instance(context, createInfo);
        }

        void createSurface() {
            VkSurfaceKHR _surface;
            if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
                throw std::runtime_error("failed to create window surface");
            }
            surface = vk::raii::SurfaceKHR(instance, _surface);
        }

        void pickPhysicalDevice() {
            std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();

            if (devices.empty()) {
                throw std::runtime_error("failed to find GPUs with Vulkan support!");
            }

            for (const vk::raii::PhysicalDevice& device : devices) {
                vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
                const char* name = deviceProperties.deviceName.data();
                physicalDevice = device;
                return;

                // vk::PhysicalDeviceFeatures deviceFeatures = device.getFeatures();
                //
                // const char* name = deviceProperties.deviceName.data();
                //
                // if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu && deviceFeatures.geometryShader) {
                //     std::println("Using {}", name);
                //     physicalDevice = device;
                //     return;
                // }
            }

            throw std::runtime_error("failed to find a suitable GPU");
        }

        void findQueueFamilies() {
            std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
            bool graphicsIndexSet = false;
            bool presentationIndexSet = false;

            for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
                if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                    queueFamilyIndices.graphicsIndex = i;
                    graphicsIndexSet = true;
                }
                VkBool32 presentSupport = physicalDevice.getSurfaceSupportKHR(i, *surface);
                if (presentSupport) {
                    queueFamilyIndices.presentationIndex = i;
                    presentationIndexSet = true;
                }
                if (graphicsIndexSet && presentationIndexSet) {
                    return;
                }
            }
            
            throw std::runtime_error("failed to find queue family that supports graphics");
        }

        void createLogicalDevice() {
            float queuePriority = 0.5f;
            vk::DeviceQueueCreateInfo deviceQueueCreateInfo {
                .queueFamilyIndex = queueFamilyIndices.graphicsIndex,
                .queueCount = 1,
                .pQueuePriorities = &queuePriority,
            };

            vk::StructureChain<
                vk::PhysicalDeviceFeatures2, 
                vk::PhysicalDeviceVulkan11Features, 
                vk::PhysicalDeviceVulkan13Features, 
                vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
            > featureChain = {
                {},
                { .shaderDrawParameters = true },
                { .synchronization2 = true, .dynamicRendering = true },
                { .extendedDynamicState = true }
            };

            vk::DeviceCreateInfo deviceCreateInfo{
                .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
                .queueCreateInfoCount = 1,
                .pQueueCreateInfos = &deviceQueueCreateInfo,
                .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
                .ppEnabledExtensionNames = deviceExtensions.data()
            };

            device = vk::raii::Device(physicalDevice, deviceCreateInfo);
        }

        void getQueues() {
            graphicsQueue = device.getQueue(queueFamilyIndices.graphicsIndex, 0);
            presentQueue = device.getQueue(queueFamilyIndices.presentationIndex, 0);
        }

        void createSwapchain() {
            vk::SurfaceFormatKHR format = chooseSwapSurfaceFormat();
            vk::PresentModeKHR presentMode = chooseSwapPresentMode();
            vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
            vk::Extent2D extent = chooseSwapExtent(surfaceCapabilities);

            auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
            if (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) {
                minImageCount = surfaceCapabilities.maxImageCount;
            }

            vk::SwapchainCreateInfoKHR swapchainCreateInfo{
                .flags = vk::SwapchainCreateFlagsKHR(),
                .surface = *surface,
                .minImageCount = minImageCount,
                .imageFormat = format.format,
                .imageColorSpace = format.colorSpace,
                .imageExtent = extent,
                .imageArrayLayers = 1,
                .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
                .preTransform = surfaceCapabilities.currentTransform,
                .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                .presentMode = presentMode,
                .clipped = true,
                .oldSwapchain = nullptr
            };

            if (queueFamilyIndices.graphicsIndex != queueFamilyIndices.presentationIndex) {
                uint32_t queueFamilyIndicesArray[] = { queueFamilyIndices.graphicsIndex, queueFamilyIndices.presentationIndex };
                swapchainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
                swapchainCreateInfo.queueFamilyIndexCount = 2;
                swapchainCreateInfo.pQueueFamilyIndices = queueFamilyIndicesArray;
            } else {
                swapchainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
            }

            swapchain = vk::raii::SwapchainKHR(device, swapchainCreateInfo);
            swapchainImages = swapchain.getImages();
            swapchainImageFormat = format.format;
            swapchainExtent = extent;
        }

        void recreateSwapchain() {
            device.waitIdle();

            cleanupSwapchain();

            createSwapchain();
            createImageViews();
        }

        void cleanupSwapchain() {
            swapchainImageViews.clear();
            swapchain = nullptr;
        }

        void createImageViews() {
            swapchainImageViews.clear();

            vk::ImageViewCreateInfo imageViewCreateInfo{
                .viewType = vk::ImageViewType::e2D,
                .format = swapchainImageFormat,
                .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                }
            };

            for (vk::Image image : swapchainImages) {
                imageViewCreateInfo.image = image;
                swapchainImageViews.emplace_back(device, imageViewCreateInfo);
            }
        }

        void createGraphicsPipeline() {
            std::vector<char> vertShaderCode = readFile("build/shaders/shader_vertMain.spv");
            vk::raii::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);

            std::vector<char> fragShaderCode = readFile("build/shaders/shader_fragMain.spv");
            vk::raii::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);
            
            vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = vertShaderModule,
                .pName = "vertMain"
            };

            vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = fragShaderModule,
                .pName = "fragMain"
            };

            vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

            vk::PipelineVertexInputStateCreateInfo vertexInputInfo;

            vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
                .topology = vk::PrimitiveTopology::eTriangleList
            };

            std::vector<vk::DynamicState> dynamicStates = {
                vk::DynamicState::eViewport,
                vk::DynamicState::eScissor
            };

            vk::PipelineViewportStateCreateInfo viewportState{
                .viewportCount = 1,
                .scissorCount = 1
            };

            vk::PipelineDynamicStateCreateInfo dynamicState{
                .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
                .pDynamicStates = dynamicStates.data()
            };

            vk::PipelineRasterizationStateCreateInfo rasterizer{
                .depthClampEnable = vk::False,
                .rasterizerDiscardEnable = vk::False,
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eBack,
                .frontFace = vk::FrontFace::eClockwise,
                .depthBiasEnable = vk::False,
                .depthBiasSlopeFactor = 1.0f,
                .lineWidth = 1.0f
            };

            vk::PipelineMultisampleStateCreateInfo multisampling{
                .rasterizationSamples = vk::SampleCountFlagBits::e1,
                .sampleShadingEnable = vk::False
            };

            vk::PipelineColorBlendAttachmentState colorBlendAttachment{
                .blendEnable = vk::False,
                .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
            };

            vk::PipelineColorBlendStateCreateInfo colorBlending{
                .logicOpEnable = vk::False,
                .logicOp = vk::LogicOp::eCopy,
                .attachmentCount = 1,
                .pAttachments = &colorBlendAttachment
            };

            vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
                .setLayoutCount = 0,
                .pushConstantRangeCount = 0
            };

            pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

            vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapchainImageFormat,
            };

            vk::GraphicsPipelineCreateInfo pipelineInfo{
                .pNext = &pipelineRenderingCreateInfo,
                .stageCount = 2,
                .pStages = shaderStages,
                .pVertexInputState = &vertexInputInfo,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pColorBlendState = &colorBlending,
                .pDynamicState = &dynamicState,
                .layout = pipelineLayout,
                .renderPass = nullptr
            };

            graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
        }

        void createCommandPool() {
            vk::CommandPoolCreateInfo poolInfo{
                .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                .queueFamilyIndex = queueFamilyIndices.graphicsIndex
            };

            commandPool = vk::raii::CommandPool(device, poolInfo);
        }

        void createCommandBuffers() {
            vk::CommandBufferAllocateInfo allocInfo{
                .commandPool = commandPool,
                .level = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = MAX_FRAMES_IN_FLIGHT
            };

            commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
        }

        void createSyncObjects() {
            assert(renderFinishedSemaphores.empty() && presentCompleteSemaphores.empty() && drawFences.empty());

            for (size_t i = 0; i < swapchainImages.size(); i++) {
                renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
            }

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
                drawFences.emplace_back(device, vk::FenceCreateInfo{ .flags = vk::FenceCreateFlagBits::eSignaled });
            }
        }

        // Helper functions
        void transitionImageLayout(
            uint32_t imageIndex,
            vk::ImageLayout oldLayout,
            vk::ImageLayout newLayout,
            vk::AccessFlags2 srcAccessMask,
            vk::AccessFlags2 dstAccessMask,
            vk::PipelineStageFlags2 srcStageMask,
            vk::PipelineStageFlags2 dstStageMask
        ) { 
            vk::ImageMemoryBarrier2 barrier = {
                .srcStageMask = srcStageMask,
                .srcAccessMask = srcAccessMask,
                .dstStageMask = dstStageMask,
                .dstAccessMask = dstAccessMask,
                .oldLayout = oldLayout,
                .newLayout = newLayout,
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .image = swapchainImages[imageIndex],
                .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                }
            };

            vk::DependencyInfo dependencyInfo = {
                .dependencyFlags = {},
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &barrier
            };

            commandBuffers[frameIndex].pipelineBarrier2(dependencyInfo);
        }

        std::vector<const char*> getRequiredExtentions() {
            uint32_t glfwRequiredExtensionCount = 0;
            const char** glfwRequiredExtensions = glfwGetRequiredInstanceExtensions(&glfwRequiredExtensionCount);

            std::vector<const char*> extensions(glfwRequiredExtensions, glfwRequiredExtensions + glfwRequiredExtensionCount);
            
            if (enableValidationLayers) {
                extensions.push_back(vk::EXTDebugUtilsExtensionName);
            }

            if (macOS) {
                extensions.push_back(vk::KHRPortabilityEnumerationExtensionName);
            }

            return extensions;
        } 

        std::vector<const char*> getRequiredLayers() {
            std::vector<const char*> layers;
            if (enableValidationLayers) {
                layers.assign(validationLayers.begin(), validationLayers.end());
            }

            return layers;
        }

        void ensureLayersSupported(const std::vector<const char*>& requiredLayers) {
            std::vector<vk::LayerProperties> supportedLayerProperties = context.enumerateInstanceLayerProperties();

            for (const char* requiredLayer : requiredLayers) {
                const bool notSupported = std::ranges::none_of(
                    supportedLayerProperties, 
                    [&requiredLayer](const vk::LayerProperties& layerProperty) {
                        return strcmp(layerProperty.layerName, requiredLayer) == 0;
                    }
                );

                if (notSupported) {
                    std::string message = std::format("Required layer not supported: {}", requiredLayer);
                    throw std::runtime_error(message);
                }
            }
        }

        void ensureExtensionsSupported(const std::vector<const char*>& requiredExtensions) {
            std::vector<vk::ExtensionProperties> supportedExtensionProperties = context.enumerateInstanceExtensionProperties();

            for (const char* requiredExtension : requiredExtensions) {
                const bool notSupported = std::ranges::none_of( 
                    supportedExtensionProperties, 
                    [&requiredExtension](const vk::ExtensionProperties& extensionProperty) {
                        return strcmp(extensionProperty.extensionName, requiredExtension) == 0;
                    }
                );

                if (notSupported) {
                    std::string message = std::format("Required extension not supported: {}", requiredExtension);
                    throw std::runtime_error(message);
                }
            }
        }
        
        vk::SurfaceFormatKHR chooseSwapSurfaceFormat() {
            std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(surface);
            for (const vk::SurfaceFormatKHR& format : availableFormats) {
                if (format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                    return format;
                }
            }

            return availableFormats[0];
        }

        vk::PresentModeKHR chooseSwapPresentMode() {
            std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface);
            for (const vk::PresentModeKHR& presentMode : availablePresentModes) {
                if (presentMode == vk::PresentModeKHR::eMailbox) {
                    return presentMode;
                }
            }

            return vk::PresentModeKHR::eFifo;
        }

        vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR capabilities) {
            if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
                return capabilities.currentExtent;
            }
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            return clampedExtent(capabilities, width, height);
        }

        vk::Extent2D clampedExtent(vk::SurfaceCapabilitiesKHR& capabilities, int& width, int& height) {
            return {
                std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
            };
        }

        vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
            vk::ShaderModuleCreateInfo createInfo{
                .codeSize = code.size(),
                .pCode = reinterpret_cast<const uint32_t*>(code.data())
            };
            vk::raii::ShaderModule shaderModule{ device, createInfo };
            return shaderModule;
        }

        static std::vector<char> readFile(const std::string& filename) {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("failed to open file!");
            }
            std::vector<char> buffer(file.tellg());
            file.seekg(0, std::ios::beg);
            file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            file.close();
            return buffer;
        }

        static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
            std::cerr << to_string(type) << " " << pCallbackData->pMessage << std::endl;
            return vk::False;
        }

        void setupDebugMessenger() {
            if (!enableValidationLayers) return;

            vk::DebugUtilsMessageSeverityFlagsEXT severityFlags( vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError );
            vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags( vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation );
            vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
                .messageSeverity = severityFlags,
                .messageType = messageTypeFlags,
                .pfnUserCallback = &debugCallback
            };
            debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
        }

};

int main() {
    {
        Pong pong;

        try {
            pong.run();
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    glfwTerminate();

    return EXIT_SUCCESS;
}

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <print>
#include <fstream>
#include <chrono>
#include <unordered_map>
#include <bit>

#ifndef GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_VULKAN
#endif
#include <GLFW/glfw3.h>

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/gtx/hash.hpp>

#include <stb/stb_image.h>
#include <tiny_obj_loader.h>

#ifndef VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#endif
#include <vulkan/vulkan_raii.hpp>


constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;
const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";

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

struct Window {
    GLFWwindow* handle = nullptr;
    operator GLFWwindow*() const { return handle; }
    ~Window() {
        if (handle) glfwDestroyWindow(handle);
        glfwTerminate();
    }
};

struct QueueFamilyIndices {
    uint32_t graphicsIndex;
    uint32_t presentationIndex;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
    glm::vec2 textureCoordinates;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return {
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex
        };
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription{
                .location = 0,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, position)
            },
            vk::VertexInputAttributeDescription{
                .location = 1,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, color)
            },
            vk::VertexInputAttributeDescription{
                .location = 2,
                .binding = 0,
                .format = vk::Format::eR32G32Sfloat,
                .offset = offsetof(Vertex, textureCoordinates)
            },
        };
    }

    bool operator==(const Vertex& other) const {
        return position == other.position && color == other.color && textureCoordinates == other.textureCoordinates;
    }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.position) ^ 
                   (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ 
                   (hash<glm::vec2>()(vertex.textureCoordinates) << 1);
        }
    };
}


// Explicit alignment for shader 
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};

// const std::vector<Vertex> vertices = {
//     {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
//     {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
//     {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
//     {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
//
//     {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
//     {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
//     {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
//     {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
// };
//
// const std::vector<uint16_t> indices = {
//     0, 1, 2, 2, 3, 0,
//     4, 5, 6, 6, 7, 4
// };

class Pong {
    public:
        void run() {
            initWindow();
            initVulkan();
            mainLoop();
        }

    private:
        Window window;
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
        vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
        vk::raii::PipelineLayout pipelineLayout = nullptr;
        vk::raii::Pipeline graphicsPipeline = nullptr;
        vk::raii::CommandPool commandPool = nullptr;
        std::vector<vk::raii::CommandBuffer> frameCommandBuffers;
        std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
        std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
        std::vector<vk::raii::Fence> drawFences;
        uint32_t frameIndex = 0;
        bool frameBufferResized = false;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        vk::raii::Buffer vertexBuffer = nullptr;
        vk::raii::DeviceMemory vertexBufferMemory = nullptr;
        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;
        vk::raii::Buffer indexBuffer = nullptr;
        vk::raii::DeviceMemory indexBufferMemory = nullptr;
        std::vector<vk::raii::Buffer> uniformBuffers;
        std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
        std::vector<void*> uniformBuffersMapped;
        vk::raii::DescriptorPool descriptorPool = nullptr;
        std::vector<vk::raii::DescriptorSet> descriptorSets;
        vk::raii::Image textureImage = nullptr;
        vk::raii::DeviceMemory textureImageMemory = nullptr;
        vk::raii::ImageView textureImageView = nullptr;
        vk::raii::Sampler textureSampler = nullptr;
        vk::raii::Image depthImage = nullptr;
        vk::raii::DeviceMemory depthImageMemory = nullptr;
        vk::raii::ImageView depthImageView = nullptr;
        vk::raii::Image colorImage = nullptr;
        vk::raii::DeviceMemory colorImageMemory = nullptr;
        vk::raii::ImageView colorImageView = nullptr;
        uint32_t mipLevels;
        vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;
        // mk:members
        
        void initWindow() {
            glfwInit();

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHintString(GLFW_WAYLAND_APP_ID, "game");

            window.handle = glfwCreateWindow(WIDTH, HEIGHT, "Pong", nullptr, nullptr);

            glfwSetWindowUserPointer(window, this);
            glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
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
            createSwapchainImageViews();
            createDescriptorSetLayout();
            createGraphicsPipeline();
            createCommandPool();
            createColorResources();
            createDepthResources();
            createTextureImage();
            createTextureImageView();
            createTextureSampler();
            loadModel();
            createVertexBuffer();
            createIndexBuffer();
            createUniformBuffers();
            createDescriptorPool();
            createDescriptorSets();
            createCommandBuffers();
            createSyncObjects();
        }

        void mainLoop() {
            while (true) {
                glfwPollEvents();
                if (glfwWindowShouldClose(window)) {
                    break;
                }
                drawFrame();
            }

            device.waitIdle();
        }

        void drawFrame() {
            vk::Result fenceResult = device.waitForFences(*drawFences[frameIndex], vk::True, UINT64_MAX);

            if (fenceResult != vk::Result::eSuccess) {
                throw std::runtime_error("failed to wait for fence");
            }

            auto [acquireResult, imageIndex] = swapchain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

            if (acquireResult == vk::Result::eErrorOutOfDateKHR) {
                recreateSwapchain();
                return;
            }

            updateUniformBuffer(frameIndex);

            recordFrameCommandBuffer(imageIndex);

            device.resetFences(*drawFences[frameIndex]);

            vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
            const vk::SubmitInfo submitInfo{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
                .pWaitDstStageMask = &waitDestinationStageMask,
                .commandBufferCount = 1,
                .pCommandBuffers = &*frameCommandBuffers[frameIndex],
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
            };

            graphicsQueue.submit(submitInfo, *drawFences[frameIndex]);

            // - Wait semaphore is not reset until operation completes
            // - The only way we know an image is done presenting (and therefore renderFinishedSemaphore reset) 
            //   is if acquireNextImage returns that imageIndex 
            // - Therefore, we need to use renderFinishedSemaphores with imageIndex to be sure we're not using it before it's reset
            const vk::PresentInfoKHR presentInfo{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
                .swapchainCount = 1,
                .pSwapchains = &*swapchain,
                .pImageIndices = &imageIndex
            };

            vk::Result presentResult = presentQueue.presentKHR(presentInfo);

            if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR || frameBufferResized) {
                frameBufferResized = false;
                recreateSwapchain();
            }

            frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
        }
        
        void updateUniformBuffer(uint32_t currentFrameIndex) {
            static std::chrono::time_point startTime = std::chrono::high_resolution_clock::now();

            std::chrono::time_point currentTime = std::chrono::high_resolution_clock::now();
            float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

            UniformBufferObject ubo{};
            ubo.model = glm::rotate(glm::mat4(1.0f), time*glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            ubo.projection = glm::perspective(glm::radians(45.0f), static_cast<float>(swapchainExtent.width) / static_cast<float>(swapchainExtent.height), 0.1f, 10.0f);
            ubo.projection[1][1] *= -1;
            memcpy(uniformBuffersMapped[currentFrameIndex], &ubo, sizeof(ubo));
        }

        void recordFrameCommandBuffer(uint32_t imageIndex) {
            vk::raii::CommandBuffer& commandBuffer = frameCommandBuffers[frameIndex];

            commandBuffer.begin({});

            recordImageLayoutTransition(
                commandBuffer,
                swapchainImages[imageIndex],
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal,
                {},
                vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::ImageAspectFlagBits::eColor,
                1
            );

            recordImageLayoutTransition(
                commandBuffer,
                colorImage,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal,
                {},
                vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::ImageAspectFlagBits::eColor,
                1
            );

            recordImageLayoutTransition(
                commandBuffer,
                depthImage,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eDepthAttachmentOptimal,
                vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
                vk::ImageAspectFlagBits::eDepth,
                1
            );

            vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
            vk::ClearDepthStencilValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

            vk::RenderingAttachmentInfo colorAttachmentInfo = {
                .imageView = colorImageView,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .resolveMode = vk::ResolveModeFlagBits::eAverage,
                .resolveImageView = swapchainImageViews[imageIndex],
                .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                // load/store ops are for primary image (multisample) - resolved image is stored implicitly
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eDontCare,
                .clearValue = clearColor
            };

            vk::RenderingAttachmentInfo depthAttachmentInfo = {
                .imageView = depthImageView,
                .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eDontCare,
                .clearValue = clearDepth,
            };

            vk::RenderingInfo renderingInfo = {
                .renderArea = { .offset = { 0, 0 }, .extent = swapchainExtent },
                .layerCount = 1,
                .colorAttachmentCount = 1,
                .pColorAttachments = &colorAttachmentInfo,
                .pDepthAttachment = &depthAttachmentInfo
            };

            commandBuffer.beginRendering(renderingInfo);
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
            commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
            commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[frameIndex], nullptr);

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
            commandBuffer.drawIndexed(indices.size(), 1, 0, 0, 0);
            commandBuffer.endRendering();

            recordImageLayoutTransition(
                commandBuffer,
                swapchainImages[imageIndex],
                vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageLayout::ePresentSrcKHR,
                vk::AccessFlagBits2::eColorAttachmentWrite,
                {},
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::PipelineStageFlagBits2::eBottomOfPipe,
                vk::ImageAspectFlagBits::eColor,
                1
            );

            commandBuffer.end();
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
                msaaSamples = getMaxSampleCount();
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
                { .features = { .samplerAnisotropy = true } },
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
            // Pause if window is minimized
            int width = 0, height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            while (width == 0 || height == 0) {
                glfwGetFramebufferSize(window, &width, &height);
                glfwWaitEvents();
            }

            device.waitIdle();

            // Cleanup
            swapchainImageViews.clear();
            presentCompleteSemaphores.clear();
            renderFinishedSemaphores.clear();
            drawFences.clear();
            swapchain = nullptr;

            // Recreate
            createSwapchain();
            createSwapchainImageViews();
            createColorResources();
            createDepthResources();
            createSyncObjects();
        }

        void createSwapchainImageViews() {
            swapchainImageViews.clear();

            for (vk::Image image : swapchainImages) {
                swapchainImageViews.push_back(createImageView(image, swapchainImageFormat, vk::ImageAspectFlagBits::eColor, 1));
            }
        }

        void createDescriptorSetLayout() {
            std::array bindings = {
                vk::DescriptorSetLayoutBinding{
                    .binding = 0,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .descriptorCount = 1,
                    .stageFlags = vk::ShaderStageFlagBits::eVertex
                },
                vk::DescriptorSetLayoutBinding{
                    .binding = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .descriptorCount = 1,
                    .stageFlags = vk::ShaderStageFlagBits::eFragment
                }
            };

            vk::DescriptorSetLayoutCreateInfo layoutInfo{
                .bindingCount = static_cast<uint32_t>(bindings.size()),
                .pBindings = bindings.data()
            };

            descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
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

            vk::VertexInputBindingDescription bindingDescription = Vertex::getBindingDescription();
            std::array attributeDescriptions = Vertex::getAttributeDescriptions();

            vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &bindingDescription,
                .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
                .pVertexAttributeDescriptions = attributeDescriptions.data()
            };

            vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
                .topology = vk::PrimitiveTopology::eTriangleList
            };

            std::array dynamicStates = {
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
                .frontFace = vk::FrontFace::eCounterClockwise,
                .depthBiasEnable = vk::False,
                .depthBiasSlopeFactor = 1.0f,
                .lineWidth = 1.0f
            };

            vk::PipelineMultisampleStateCreateInfo multisampling{
                .rasterizationSamples = msaaSamples,
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
                .setLayoutCount = 1,
                .pSetLayouts = &*descriptorSetLayout,
                .pushConstantRangeCount = 0
            };

            pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

            vk::PipelineDepthStencilStateCreateInfo depthStencil{
                .depthTestEnable = vk::True,
                .depthWriteEnable = vk::True,
                .depthCompareOp = vk::CompareOp::eLess,
                .depthBoundsTestEnable = vk::False,
                .stencilTestEnable = vk::False
            };

            vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapchainImageFormat,
                .depthAttachmentFormat = vk::Format::eD32Sfloat
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
                .pDepthStencilState = &depthStencil,
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

        void createColorResources() {
            createImage(
                swapchainExtent.width, 
                swapchainExtent.height, 
                1, 
                msaaSamples,
                swapchainImageFormat,
                vk::ImageUsageFlagBits::eColorAttachment,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                colorImage,
                colorImageMemory
            );
            colorImageView = createImageView(colorImage, swapchainImageFormat, vk::ImageAspectFlagBits::eColor, 1);
        }

        void createDepthResources() {
            vk::Format depthFormat = vk::Format::eD32Sfloat;
            createImage(
                swapchainExtent.width,
                swapchainExtent.height,
                1,
                msaaSamples,
                depthFormat,
                vk::ImageUsageFlagBits::eDepthStencilAttachment,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                depthImage,
                depthImageMemory
            );
            depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
        }

        void createTextureImage() {
            int texWidth, texHeight, texChannels;
            stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
            vk::DeviceSize imageSize = texWidth * texHeight * 4;

            if (!pixels) {
                throw std::runtime_error("failed to load texture image!");
            }

            mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

            vk::raii::Buffer stagingBuffer = nullptr;
            vk::raii::DeviceMemory stagingBufferMemory = nullptr;

            createBuffer(
                imageSize, 
                vk::BufferUsageFlagBits::eTransferSrc, 
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                stagingBuffer,
                stagingBufferMemory
            );

            void* data = stagingBufferMemory.mapMemory(0, imageSize);
            memcpy(data, pixels, imageSize);
            stagingBufferMemory.unmapMemory();
            stbi_image_free(pixels);

            vk::Format textureImageFormat = vk::Format::eR8G8B8A8Srgb;

            createImage(
                texWidth, 
                texHeight,
                mipLevels,
                vk::SampleCountFlagBits::e1,
                textureImageFormat,
                vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, 
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                textureImage,
                textureImageMemory
            );
            
            vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
            recordImageLayoutTransition(
                commandBuffer,
                textureImage,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eTransferDstOptimal,
                {},
                vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eTopOfPipe,
                vk::PipelineStageFlagBits2::eTransfer,
                vk::ImageAspectFlagBits::eColor,
                mipLevels
            );
            recordBufferImageCopy(commandBuffer, stagingBuffer, textureImage, texWidth, texHeight);
            recordMipmapBlits(commandBuffer, textureImage, textureImageFormat, texWidth, texHeight, mipLevels);
            endSingleTimeCommands(commandBuffer);
        }

        void createTextureImageView() {
            textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mipLevels);
        }

        void createTextureSampler() {
            vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
            vk::SamplerCreateInfo samplerInfo{
                .magFilter = vk::Filter::eLinear,
                .minFilter = vk::Filter::eLinear,
                .mipmapMode = vk::SamplerMipmapMode::eLinear,
                .addressModeU = vk::SamplerAddressMode::eRepeat,
                .addressModeV = vk::SamplerAddressMode::eRepeat,
                .addressModeW = vk::SamplerAddressMode::eRepeat,
                .mipLodBias = 0.0f,
                .anisotropyEnable = vk::True,
                .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
                .compareEnable = vk::False,
                .compareOp = vk::CompareOp::eAlways,
                .minLod = 0.0f,
                .maxLod = vk::LodClampNone,
                .borderColor = vk::BorderColor::eIntOpaqueBlack,
                .unnormalizedCoordinates = vk::False
            };
            textureSampler = vk::raii::Sampler(device, samplerInfo);
        }

        void loadModel() {
            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;
            std::string warn, err;

            if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
                throw std::runtime_error(warn + err);
            }

            std::unordered_map<Vertex, uint32_t> uniqueVertices{};

            for (const auto& shape : shapes) {
                for (const auto& index : shape.mesh.indices) {
                    Vertex vertex{};

                    vertex.position = {
                        attrib.vertices[3 * index.vertex_index + 0],
                        attrib.vertices[3 * index.vertex_index + 1],
                        attrib.vertices[3 * index.vertex_index + 2],
                    };

                    vertex.textureCoordinates = {
                        attrib.texcoords[2 * index.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * index.texcoord_index + 1],
                    };

                    vertex.color = {1.0f, 1.0f, 1.0f};

                    if (uniqueVertices.count(vertex) == 0) {
                        uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                        vertices.push_back(vertex);
                    }

                    indices.push_back(uniqueVertices[vertex]);
                }
            }
        }

        void createVertexBuffer() {
            vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

            createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                stagingBuffer,
                stagingBufferMemory
            );

            void* data = stagingBufferMemory.mapMemory(0, bufferSize);
            memcpy(data, vertices.data(), bufferSize);
            stagingBufferMemory.unmapMemory();

            createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                vertexBuffer,
                vertexBufferMemory
            );

            copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
        }

        void createIndexBuffer() {
            vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

            createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                stagingBuffer,
                stagingBufferMemory
            );

            void* data = stagingBufferMemory.mapMemory(0, bufferSize);
            memcpy(data, indices.data(), bufferSize);
            stagingBufferMemory.unmapMemory();

            createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                indexBuffer,
                indexBufferMemory
            );

            copyBuffer(stagingBuffer, indexBuffer, bufferSize);
        }

        void createUniformBuffers() {
            uniformBuffers.clear();
            uniformBuffersMemory.clear();
            uniformBuffersMapped.clear();

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
                vk::raii::Buffer buffer = nullptr;
                vk::raii::DeviceMemory bufferMemory = nullptr;
                createBuffer(
                    bufferSize, 
                    vk::BufferUsageFlagBits::eUniformBuffer, 
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                    buffer,
                    bufferMemory
                );
                uniformBuffers.emplace_back(std::move(buffer));
                uniformBuffersMemory.emplace_back(std::move(bufferMemory));
                uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));
            }
        }

        void createDescriptorPool() {
            std::array poolSizes {
                vk::DescriptorPoolSize{
                    .type = vk::DescriptorType::eUniformBuffer,
                    .descriptorCount = MAX_FRAMES_IN_FLIGHT,
                },
                vk::DescriptorPoolSize{
                    .type = vk::DescriptorType::eCombinedImageSampler,
                    .descriptorCount = MAX_FRAMES_IN_FLIGHT,
                },
            };

            vk::DescriptorPoolCreateInfo poolInfo{
                .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                .maxSets = MAX_FRAMES_IN_FLIGHT,
                .poolSizeCount = poolSizes.size(),
                .pPoolSizes = poolSizes.data()
            };

            descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
        }

        void createDescriptorSets() {
            std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
            vk::DescriptorSetAllocateInfo allocInfo{
                .descriptorPool = descriptorPool,
                .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
                .pSetLayouts = layouts.data()
            };
            descriptorSets.clear();
            descriptorSets = device.allocateDescriptorSets(allocInfo);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vk::DescriptorBufferInfo bufferInfo{
                    .buffer = uniformBuffers[i],
                    .offset = 0,
                    .range = sizeof(UniformBufferObject)
                };
                vk::DescriptorImageInfo imageInfo{
                    .sampler = textureSampler,
                    .imageView = textureImageView,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
                };
                std::array descriptorWrites{
                    vk::WriteDescriptorSet{
                        .dstSet = descriptorSets[i],
                        .dstBinding = 0,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eUniformBuffer,
                        .pBufferInfo = &bufferInfo
                    },
                    vk::WriteDescriptorSet{
                        .dstSet = descriptorSets[i],
                        .dstBinding = 1,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                        .pImageInfo = &imageInfo
                    },
                };
                device.updateDescriptorSets(descriptorWrites, {});
            }
        }

        void createCommandBuffers() {
            vk::CommandBufferAllocateInfo allocInfo{
                .commandPool = commandPool,
                .level = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = MAX_FRAMES_IN_FLIGHT
            };

            frameCommandBuffers = vk::raii::CommandBuffers(device, allocInfo);
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
        vk::SampleCountFlagBits getMaxSampleCount() {
            vk::PhysicalDeviceProperties props = physicalDevice.getProperties();
            vk::SampleCountFlags counts = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
            return static_cast<vk::SampleCountFlagBits>(std::bit_floor(static_cast<uint32_t>(counts))); 
        }

        void recordMipmapBlits(
            vk::raii::CommandBuffer& commandBuffer,
            vk::raii::Image& image, 
            vk::Format imageFormat, 
            int32_t texWidth, 
            int32_t texHeight, uint32_t 
            mipLevels
        ) {
            vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(imageFormat);
            if (!(vk::FormatFeatureFlagBits::eSampledImageFilterLinear & formatProperties.optimalTilingFeatures)) {
                throw std::runtime_error("texture image format does not support linear blitting");
            }

            // Partial barrier to be used multiple times
            vk::ImageMemoryBarrier2 barrier = {
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .image = image,
                .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
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

            int32_t mipWidth = texWidth;
            int32_t mipHeight = texHeight;

            for (uint32_t i = 1; i < mipLevels; i++) {
                int32_t nextMipWidth = mipWidth > 1 ? mipWidth/2 : 1;
                int32_t nextMipHeight = mipHeight > 1 ? mipHeight/2 : 1;

                // First barrier transitions mipLevel eTransferDstOptimal->eTransferSrcOptimal, 
                // and waits on write from buffer copy (first iteration) or previous blit
                barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
                barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
                barrier.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
                barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
                barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
                barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
                barrier.subresourceRange.baseMipLevel = i - 1;
                commandBuffer.pipelineBarrier2(dependencyInfo);

                vk::ArrayWrapper1D<vk::Offset3D, 2> offsets, dstOffsets;
                offsets[0] = vk::Offset3D(0, 0, 0);
                offsets[1] = vk::Offset3D(mipWidth, mipHeight, 1);
                dstOffsets[0] = vk::Offset3D(0, 0, 0);
                dstOffsets[1] = vk::Offset3D(nextMipWidth, nextMipHeight, 1);

                vk::ImageSubresourceLayers srcSubresource{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = i - 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                };

                vk::ImageSubresourceLayers dstSubresource{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = i,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                };

                vk::ImageBlit blit{
                    .srcSubresource = srcSubresource,
                    .srcOffsets = offsets,
                    .dstSubresource = dstSubresource,
                    .dstOffsets = dstOffsets
                };

                commandBuffer.blitImage(
                    image, vk::ImageLayout::eTransferSrcOptimal,
                    image, vk::ImageLayout::eTransferDstOptimal,
                    { blit }, vk::Filter::eLinear
                );

                // Second barrier transitions mipLevel to eShaderReadOnlyOptimal, and waits on
                // reads from the blit to finish
                barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
                barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
                barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
                barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
                barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
                barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

                commandBuffer.pipelineBarrier2(dependencyInfo);

                mipWidth = nextMipWidth;
                mipHeight = nextMipHeight;
            }

            // Finally, transition the last mipLevel from eTransferDstOptimal->eShaderReadOnlyOptimal.
            // It was never transitioned to src because we didn't need to read from it. Wait on the 
            // writes from the blit.
            barrier.subresourceRange.baseMipLevel = mipLevels - 1;
            barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
            barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
            barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
            barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
            barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

            commandBuffer.pipelineBarrier2(dependencyInfo);
        }

        [[nodiscard]] vk::raii::ImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspectMask, uint32_t mipLevels) {
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

            return vk::raii::ImageView(device, createInfo);
        }

        vk::raii::CommandBuffer beginSingleTimeCommands() {
            vk::CommandBufferAllocateInfo allocInfo{
                .commandPool = commandPool,
                .level = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = 1
            };
            vk::raii::CommandBuffer commandBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
            vk::CommandBufferBeginInfo beginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };

            commandBuffer.begin(beginInfo);

            return commandBuffer;
        }

        void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer) {
            commandBuffer.end();

            vk::SubmitInfo submitInfo{
                .commandBufferCount = 1,
                .pCommandBuffers = &*commandBuffer
            };

            graphicsQueue.submit(submitInfo, nullptr);

            device.waitIdle();
        }

        void createImage(
            uint32_t width, 
            uint32_t height, 
            uint32_t mipLevels,
            vk::SampleCountFlagBits numSamples,
            vk::Format format, 
            vk::ImageUsageFlags usage, 
            vk::MemoryPropertyFlags properties, 
            vk::raii::Image& image, 
            vk::raii::DeviceMemory& imageMemory
        ) {
            vk::ImageCreateInfo imageInfo{
                .imageType = vk::ImageType::e2D,
                .format = format,
                .extent = { width, height, 1 },
                .mipLevels = mipLevels,
                .arrayLayers = 1,
                .samples = numSamples,
                .tiling = vk::ImageTiling::eOptimal,
                .usage = usage,
                .sharingMode = vk::SharingMode::eExclusive,
                .initialLayout = vk::ImageLayout::eUndefined
            };

            image = vk::raii::Image(device, imageInfo);

            vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
            vk::MemoryAllocateInfo allocInfo{
                .allocationSize = memRequirements.size,
                .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
            };
            imageMemory = vk::raii::DeviceMemory(device, allocInfo);
            image.bindMemory(imageMemory, 0);
        }

        void copyBuffer(
            vk::raii::Buffer& srcBuffer,
            vk::raii::Buffer& dstBuffer,
            vk::DeviceSize size
        ) {
            // Cloud use a separate transfer queue
            // Cloud use a transient command buffer
            vk::raii::CommandBuffer copyCommandBuffer = beginSingleTimeCommands();
            copyCommandBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
            endSingleTimeCommands(copyCommandBuffer);
        }

        void createBuffer(
            vk::DeviceSize size, 
            vk::BufferUsageFlags usage, 
            vk::MemoryPropertyFlags properties, 
            vk::raii::Buffer& buffer, 
            vk::raii::DeviceMemory& bufferMemory
        ) {
            vk::BufferCreateInfo bufferInfo{
                .size = size,
                .usage = usage,
                .sharingMode = vk::SharingMode::eExclusive
            };

            buffer = vk::raii::Buffer(device, bufferInfo);

            vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

            vk::MemoryAllocateInfo allocInfo{
                .allocationSize = memRequirements.size,
                .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
            };

            bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
            // Real world app would bind multiple buffers to a single large allocation 
            // using offsets because devices have simultanious allocation limits
            // You can even put multiple vertex/index buffers in the smae VkBuffer (which driver developers recommend)
            buffer.bindMemory(*bufferMemory, 0);
        }

        uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
            vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

            for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
                if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                    return i;
                }
            }

            throw std::runtime_error("failed to find suitable memory type!");
        }

        void recordImageLayoutTransition(
            vk::raii::CommandBuffer& commandBuffer,
            vk::Image image,
            vk::ImageLayout oldLayout,
            vk::ImageLayout newLayout,
            vk::AccessFlags2 srcAccessMask,
            vk::AccessFlags2 dstAccessMask,
            vk::PipelineStageFlags2 srcStageMask,
            vk::PipelineStageFlags2 dstStageMask,
            vk::ImageAspectFlagBits imageAspectMask,
            uint32_t mipLevels
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
                .image = image,
                .subresourceRange = {
                    .aspectMask = imageAspectMask,
                    .baseMipLevel = 0,
                    .levelCount = mipLevels,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                }
            };

            vk::DependencyInfo dependencyInfo = {
                .dependencyFlags = {},
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &barrier
            };

            commandBuffer.pipelineBarrier2(dependencyInfo);
        }

        void recordBufferImageCopy(
            vk::raii::CommandBuffer& commandBuffer,
            vk::Buffer buffer,
            vk::Image image,
            uint32_t width,
            uint32_t height
        ) {
            vk::BufferImageCopy region{
                .bufferOffset = 0,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                },
                .imageOffset = { 0, 0, 0 },
                .imageExtent = { width, height, 1 }
            };

            commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});
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

        static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
            Pong* app = reinterpret_cast<Pong*>(glfwGetWindowUserPointer(window));
            app->frameBufferResized = true;
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
    Pong pong;

    try {
        pong.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <print>
#include <fstream>
#include <chrono>

#ifndef GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_VULKAN
#endif
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

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
    glm::vec2 position;
    glm::vec3 color;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return {
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex
        };
    }

    static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription{
                .location = 0,
                .binding = 0,
                .format = vk::Format::eR32G32Sfloat,
                .offset = offsetof(Vertex, position)
            },
            vk::VertexInputAttributeDescription{
                .location = 1,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, color)
            },
        };
    }
};

// Explicit alignment for shader 
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
};

const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0
};

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
        std::vector<vk::raii::CommandBuffer> commandBuffers;
        std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
        std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
        std::vector<vk::raii::Fence> drawFences;
        uint32_t frameIndex = 0;
        bool frameBufferResized = false;
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
            createImageViews();
            createDescriptorSetLayout();
            createGraphicsPipeline();
            createCommandPool();
            createTextureImage();
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

            if (acquireResult == vk::Result::eSuboptimalKHR) {
                std::println("acquireResult suboptimal");
            }

            updateUniformBuffer(frameIndex);

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
            commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
            commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);
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
            createImageViews();
            createSyncObjects();
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

        void createDescriptorSetLayout() {
            vk::DescriptorSetLayoutBinding uboLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex
            };
            vk::DescriptorSetLayoutCreateInfo layoutInfo{
                .bindingCount = 1,
                .pBindings = &uboLayoutBinding
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
            std::vector<vk::VertexInputAttributeDescription> attributeDescriptions = Vertex::getAttributeDescriptions();

            vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &bindingDescription,
                .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
                .pVertexAttributeDescriptions = attributeDescriptions.data()
            };

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
                .frontFace = vk::FrontFace::eCounterClockwise,
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
                .setLayoutCount = 1,
                .pSetLayouts = &*descriptorSetLayout,
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

        void createTextureImage() {
            int texWidth, texHeight, texChannels;
            stbi_uc* pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
            vk::DeviceSize imageSize = texWidth * texHeight * 4;

            if (!pixels) {
                throw std::runtime_error("failed to load texture image!");
            }

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
            vk::DescriptorPoolSize poolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT);
            vk::DescriptorPoolCreateInfo poolInfo{
                .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                .maxSets = MAX_FRAMES_IN_FLIGHT,
                .poolSizeCount = 1,
                .pPoolSizes = &poolSize
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
                vk::WriteDescriptorSet descriptorWrite{
                    .dstSet = descriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &bufferInfo
                };
                device.updateDescriptorSets(descriptorWrite, {});
            }

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
        void copyBuffer(
            vk::raii::Buffer& srcBuffer,
            vk::raii::Buffer& dstBuffer,
            vk::DeviceSize size
        ) {
            // Cloud use a separate transfer queue
            // Cloud use a transient command buffer
            vk::CommandBufferAllocateInfo allocInfo{
                .commandPool = commandPool,
                .level = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = 1
            };
            vk::raii::CommandBuffer copyCommandBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
            copyCommandBuffer.begin(vk::CommandBufferBeginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
            copyCommandBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
            copyCommandBuffer.end();
            graphicsQueue.submit(vk::SubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*copyCommandBuffer }, nullptr);
            graphicsQueue.waitIdle();
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

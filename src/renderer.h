#pragma once

#include "window.h"
#include "gamestate.h"

#include <cstdint>
#include <string>
#include <vector>
#include <array>

#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

constexpr int MAX_FRAMES_IN_FLIGHT = 2;
inline const std::string MODEL_PATH = "models/viking_room.obj";
inline const std::string TEXTURE_PATH = "textures/viking_room.png";

inline const std::vector<const char*> validationLayers = {
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

struct RenderEntity {
    int32_t vertexOffset;
    uint32_t vertexCount;
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t firstInstance;
    uint32_t instanceCount;
};

struct RenderInstance {
    glm::mat4 modelMatrix;
};

struct RenderState {
    std::unordered_map<int, RenderEntity> entities;
    std::vector<RenderInstance> instances;
};

class Renderer {
public:
    Renderer(Window& window, GameState& gameState);
    ~Renderer();
    void drawFrame(RenderState& renderState);

private:
    Window& window;
    GameState& gameState;
    RenderState renderState;
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
    vk::raii::PipelineLayout graphicsPipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> frameCommandBuffers;
    std::vector<vk::raii::Fence> drawFences;
    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderCompleteSemaphores;
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

    void createRenderState();
    void initVulkan();
    void updateUniformBuffer(uint32_t currentFrameIndex);
    void recordFrameCommandBuffer(uint32_t imageIndex);

    void createInstance();
    void createSurface();
    void pickPhysicalDevice();
    void findQueueFamilies();
    void createLogicalDevice();
    void getQueues();
    void createSwapchain();
    void recreateSwapchain();
    void createSwapchainImageViews();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createCommandPool();
    void createColorResources();
    void createDepthResources();
    void createTextureImage();
    void createTextureImageView();
    void createTextureSampler();
    void loadModel(std::string& path);
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void createSyncObjects();

    vk::SampleCountFlagBits getMaxSampleCount();
    void recordMipmapBlits(
        vk::raii::CommandBuffer& commandBuffer,
        vk::raii::Image& image,
        vk::Format imageFormat,
        int32_t texWidth,
        int32_t texHeight,
        uint32_t mipLevels
    );
    [[nodiscard]] vk::raii::ImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspectMask, uint32_t mipLevels);
    vk::raii::CommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer);
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
    );
    void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size);
    void createBuffer(
        vk::DeviceSize size,
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::raii::Buffer& buffer,
        vk::raii::DeviceMemory& bufferMemory
    );
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
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
    );
    void recordBufferImageCopy(
        vk::raii::CommandBuffer& commandBuffer,
        vk::Buffer buffer,
        vk::Image image,
        uint32_t width,
        uint32_t height
    );
    std::vector<const char*> getRequiredExtentions();
    std::vector<const char*> getRequiredLayers();
    void ensureLayersSupported(const std::vector<const char*>& requiredLayers);
    void ensureExtensionsSupported(const std::vector<const char*>& requiredExtensions);
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat();
    vk::PresentModeKHR chooseSwapPresentMode();
    vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR capabilities);
    vk::Extent2D clampedExtent(vk::SurfaceCapabilitiesKHR& capabilities, int& width, int& height);
    vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;

    static std::vector<char> readFile(const std::string& filename);
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
        vk::DebugUtilsMessageTypeFlagsEXT type,
        const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void*
    );
    void setupDebugMessenger();
};

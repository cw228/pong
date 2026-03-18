#pragma once

#include "window.h"
#include "context.h"
#include "render_targets.h"
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
constexpr int MAX_INSTANCES = 100;
constexpr int MAX_TEXTURES = 10;
inline const std::string TEXTURE_PATH = "textures/viking_room.png";

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
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};

struct RenderModel {
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
    std::unordered_map<uint32_t, RenderModel> models;
    std::vector<uint32_t> renderedModels;
    std::vector<RenderInstance> instances;
};

class Renderer {
public:
    Renderer(Window& window, GameState& gameState);
    ~Renderer();
    void drawFrame(GameState& gameState);

private:
    Window& window;
    GameState& gameState;
    RenderState renderState;

    VulkanContext vContext;
    RenderTargets renderTargets;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout graphicsPipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> frameCommandBuffers;
    std::vector<vk::raii::Fence> drawFences;
    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    // std::vector<vk::raii::Semaphore> renderCompleteSemaphores;
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
    std::vector<vk::raii::Buffer> storageBuffers;
    std::vector<vk::raii::DeviceMemory> storageBuffersMemory;
    std::vector<void*> storageBuffersMapped;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;
    vk::raii::Image textureImage = nullptr;
    vk::raii::DeviceMemory textureImageMemory = nullptr;
    vk::raii::ImageView textureImageView = nullptr;
    vk::raii::Sampler textureSampler = nullptr;
    uint32_t mipLevels;

    // vk::raii::Image depthImage = nullptr;
    // vk::raii::DeviceMemory depthImageMemory = nullptr;
    // vk::raii::ImageView depthImageView = nullptr;
    // vk::raii::Image colorImage = nullptr;
    // vk::raii::DeviceMemory colorImageMemory = nullptr;
    // vk::raii::ImageView colorImageView = nullptr;


    // vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;

    // mk:members

    void loadModels(GameState& gameState);
    void updateViewport();
    void updateRenderState(GameState& gameState);
    glm::mat4 createModelMatrix(Instance& instance);
    void initVulkan();
    void updateUniformBuffer(uint32_t currentFrameIndex);
    void updateStorageBuffer(uint32_t currentFrameIndex);
    void recordFrameCommandBuffer(uint32_t imageIndex);

    void recreateRenderTargets();
    // void createSwapchainImageViews();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createCommandPool();
    // void createColorResources();
    // void createDepthResources();
    void createTextureImage();
    void createTextureImageView();
    void createTextureSampler();
    void loadModel(const std::string& path);
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
    void createStorageBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void createSyncObjects();

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
    vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;

    static std::vector<char> readFile(const std::string& filename);
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
};

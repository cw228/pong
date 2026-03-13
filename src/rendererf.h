#include <vulkan/vulkan_raii.hpp>

struct RenderContext {
    vk::raii::Context context;
    vk::Instance instance;
};

#include "rendererf.h"
#include <vulkan/vulkan_raii.hpp>

constexpr int MAX_FRAMES_IN_FLIGHT = 2;
constexpr int MAX_INSTANCES = 100;

inline const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

std::vector<const char*> getRequiredLayers() {
    std::vector<const char*> layers;
    if (enableValidationLayers) {
        layers.assign(validationLayers.begin(), validationLayers.end());
    }

    return layers;
}

void ensureLayersSupported(const std::vector<const char*>& requiredLayers, const std::vector<vk::LayerProperties> layerProperties) {
    for (const char* requiredLayer : requiredLayers) {
        const bool notSupported = std::ranges::none_of(
            layerProperties,
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

std::vector<const char*> getRequiredExtentions() {
    uint32_t glfwRequiredExtensionCount = 0;
    const char** glfwRequiredExtensions = glfwGetRequiredInstanceExtensions(&glfwRequiredExtensionCount);

    std::vector<const char*> extensions(glfwRequiredExtensions, glfwRequiredExtensions + glfwRequiredExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

#ifdef __APPLE__
        extensions.push_back(vk::KHRPortabilityEnumerationExtensionName);
#endif

    return extensions;
}

void ensureExtensionsSupported(const std::vector<const char*>& requiredExtensions, const std::vector<vk::ExtensionProperties>& extensionProperties) {
    for (const char* requiredExtension : requiredExtensions) {
        const bool notSupported = std::ranges::none_of(
            extensionProperties,
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

vk::Instance createInstance(vk::raii::Context& context) {
    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName = "Pong",
        .applicationVersion = VK_MAKE_VERSION( 1, 0, 0 ),
        .pEngineName        = "No Engine",
        .engineVersion      = VK_MAKE_VERSION( 1, 0, 0 ),
        .apiVersion         = vk::ApiVersion14
    };

    std::vector<const char*> requiredLayers = getRequiredLayers();
    std::vector<vk::LayerProperties> layerProperties = context.enumerateInstanceLayerProperties();
    ensureLayersSupported(requiredLayers, layerProperties);

    std::vector<const char*> requiredExtentions = getRequiredExtentions();
    std::vector<vk::ExtensionProperties> extensionProperties = context.enumerateInstanceExtensionProperties();
    ensureExtensionsSupported(requiredExtentions, extensionProperties);

    vk::InstanceCreateInfo createInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
        .ppEnabledLayerNames = requiredLayers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(requiredExtentions.size()),
        .ppEnabledExtensionNames = requiredExtentions.data()
    };

#ifdef __APPLE__
    createInfo.flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

    return vk::raii::Instance(context, createInfo);
}


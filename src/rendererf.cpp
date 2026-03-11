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

void initRenderer() {
}

vk::Instance createInstance() {
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

#ifdef __APPLE__
    createInfo.flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

    return vk::raii::Instance(context, createInfo);
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

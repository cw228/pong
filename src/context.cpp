#include <iostream>
#include "vulkan/vulkan_raii.hpp"
#include "context.h"
#include "window.h"

std::vector<const char*> getRequiredExtentions() {
    uint32_t glfwRequiredExtensionCount = 0;
    const char** glfwRequiredExtensions = glfwGetRequiredInstanceExtensions(&glfwRequiredExtensionCount);

    std::vector<const char*> extensions(glfwRequiredExtensions, glfwRequiredExtensions + glfwRequiredExtensionCount);

#ifndef NDEBUG
        extensions.push_back(vk::EXTDebugUtilsExtensionName);
#endif

#ifdef __APPLE__
        extensions.push_back(vk::KHRPortabilityEnumerationExtensionName);
#endif

    return extensions;
}

std::vector<const char*> getRequiredLayers() {
    std::vector<const char*> layers = {
#ifndef NDEBUG
        "VK_LAYER_KHRONOS_validation"
#endif
    };
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

vk::raii::Instance createInstance(vk::raii::Context& context) {
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

VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT type,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void*
) {
    std::cerr << to_string(type) << " " << pCallbackData->pMessage << std::endl;
    return vk::False;
}

vk::raii::DebugUtilsMessengerEXT createDebugMessenger(vk::raii::Instance& instance, vk::PFN_DebugUtilsMessengerCallbackEXT callback) {
    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags( vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError );
    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags( vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation );
    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
        .messageSeverity = severityFlags,
        .messageType = messageTypeFlags,
        .pfnUserCallback = callback
    };
    return instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

vk::raii::SurfaceKHR createSurface(vk::raii::Instance& instance, Window& window) {
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
        throw std::runtime_error("failed to create window surface");
    }
    return vk::raii::SurfaceKHR(instance, _surface);
}

vk::raii::PhysicalDevice choosePhysicalDevice(vk::raii::Instance& instance) {
    std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();

    if (devices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    for (const vk::raii::PhysicalDevice& device : devices) {
        vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
        const char* name = deviceProperties.deviceName.data();
        return device;
    }

    throw std::runtime_error("failed to find a suitable GPU");
}

vk::raii::Device createLogicalDevice(vk::raii::PhysicalDevice physicalDevice, uint32_t graphicsQueueIndex) {
    std::vector<const char*> deviceExtensions = {
        vk::KHRSwapchainExtensionName,
#ifdef __APPLE__
        "VK_KHR_portability_subset",
        vk::KHRSynchronization2ExtensionName,
        vk::KHRDynamicRenderingExtensionName,
#endif
    };

    float queuePriority = 0.5f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo {
        .queueFamilyIndex = graphicsQueueIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };

    vk::StructureChain<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        vk::PhysicalDeviceTimelineSemaphoreFeatures
    > featureChain = {
        { .features = { .samplerAnisotropy = true } },
        { .shaderDrawParameters = true },
        { .synchronization2 = true, .dynamicRendering = true },
        { .extendedDynamicState = true },
        { .timelineSemaphore = true }
    };

    vk::DeviceCreateInfo deviceCreateInfo{
        .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &deviceQueueCreateInfo,
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data()
    };

    return vk::raii::Device(physicalDevice, deviceCreateInfo);
}

QueueFamilies findQueueFamilies(vk::raii::PhysicalDevice& physicalDevice, vk::raii::SurfaceKHR& surface) {
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    bool graphicsIndexSet = false;
    bool presentationIndexSet = false;

    QueueFamilies indices{};

    for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
        if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphics = i;
            graphicsIndexSet = true;
        }

        VkBool32 presentSupport = physicalDevice.getSurfaceSupportKHR(i, *surface);
        if (presentSupport) {
            indices.presentation = i;
            presentationIndexSet = true;
        }

        if (graphicsIndexSet && presentationIndexSet) {
            return indices;
        }
    }

    throw std::runtime_error("failed to find queue families");
}

VulkanContext::VulkanContext(Window& window) {
    instance = createInstance(context);
#ifndef NDEBUG
    debugMessenger = createDebugMessenger(instance, debugCallback);
#endif
    surface = createSurface(instance, window);
    physicalDevice = choosePhysicalDevice(instance);
    queueFamilies = findQueueFamilies(physicalDevice, surface);
    device = createLogicalDevice(physicalDevice, queueFamilies.graphics);
    queues.graphics = device.getQueue(queueFamilies.graphics, 0);
    queues.presentation = device.getQueue(queueFamilies.presentation, 0);
}


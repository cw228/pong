#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <print>

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
        static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
            vk::DebugUtilsMessageSeverityFlagBitsEXT severity, 
            vk::DebugUtilsMessageTypeFlagsEXT type, 
            const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, 
            void*
        ) {
            std::cerr << to_string(type) << " " << pCallbackData->pMessage << std::endl;
            return vk::False;
        }

        // mk:members
        GLFWwindow* window;
        vk::raii::Context context;
        vk::raii::Instance instance = nullptr;
        vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
        vk::raii::PhysicalDevice physicalDevice = nullptr;
        vk::raii::SurfaceKHR surface = nullptr;
        vk::raii::Device device = nullptr;
        vk::PhysicalDeviceFeatures deviceFeatures;
        std::vector<const char*> deviceExtensions = { vk::KHRSwapchainExtensionName };
        QueueFamilyIndices queueFamilyIndices;

        void initWindow() {
            glfwInit();

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

            window = glfwCreateWindow(WIDTH, HEIGHT, "Pong", nullptr, nullptr);
        }

        void initVulkan() {
            createInstance();
            setupDebugMessenger();
            createSurface();
            pickPhysicalDevice();
            findQueueFamilies();
            createLogicalDevice();
            createSwapChain();
        }

        void mainLoop() {
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
            }
        }

        void cleanup() {
            glfwDestroyWindow(window);
            glfwTerminate();
        }

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

            std::println("Created instance");
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
                std::println("Using {}", name);
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

        void createLogicalDevice() {
            float queuePriority = 0.5f;
            vk::DeviceQueueCreateInfo deviceQueueCreateInfo {
                .queueFamilyIndex = queueFamilyIndices.graphicsIndex,
                .queueCount = 1,
                .pQueuePriorities = &queuePriority,
            };

            vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featureChain = {
                {},
                { .dynamicRendering = true },
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
            // get queue with device.getQueue(graphicsQueueIndex, 0);
        }

        void createSwapChain() {
            vk::SurfaceFormatKHR format = chooseSwapSurfaceFormat();
            vk::PresentModeKHR presentMode = chooseSwapPresentMode();
            vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
            vk::Extent2D extent = chooseSwapExtent(surfaceCapabilities);
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
                    std::println("Present mode: Mailbox");
                    return presentMode;
                }
            }
            std::println("Present mode: FIFO");
            return vk::PresentModeKHR::eFifo;
        }

        vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR capabilities) {
            std::println("Swap extent: {} {}", capabilities.currentExtent.width, capabilities.currentExtent.height);
            if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
                return capabilities.currentExtent;
            }
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            std::println("Framebuffer size: {} {}", width, height);
            return {
                std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
            };
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

        void setupDebugMessenger() {
            if (!enableValidationLayers) return;

            vk::DebugUtilsMessageSeverityFlagsEXT severityFlags( vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError );
            vk::DebugUtilsMessageTypeFlagsEXT    messageTypeFlags( vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation );
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

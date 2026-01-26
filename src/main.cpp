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

class Pong {
    public:
        void run() {
            initWindow();
            initVulkan();
            mainLoop();
            cleanup();
        }

    private:
        GLFWwindow* window;
        vk::raii::Context context;
        vk::raii::Instance instance = nullptr;

        void initWindow() {
            glfwInit();

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

            window = glfwCreateWindow(WIDTH, HEIGHT, "Pong", nullptr, nullptr);
        }

        void initVulkan() {
            createInstance();
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

            std::vector<const char*> requiredLayers;
            if (enableValidationLayers) {
                requiredLayers.assign(validationLayers.begin(), validationLayers.end());
            }
            ensureLayersSupported(requiredLayers);

            uint32_t glfwRequiredExtensionCount = 0;
            const char** glfwRequiredExtensions = glfwGetRequiredInstanceExtensions(&glfwRequiredExtensionCount);
            ensureExtensionsSupported(glfwRequiredExtensions, glfwRequiredExtensionCount);

            vk::InstanceCreateInfo createInfo{
                .pApplicationInfo = &appInfo,
                .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
                .ppEnabledLayerNames = requiredLayers.data(),
                .enabledExtensionCount = glfwRequiredExtensionCount,
                .ppEnabledExtensionNames = glfwRequiredExtensions
            };

            instance = vk::raii::Instance(context, createInfo);
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

        void ensureExtensionsSupported(const char**& requiredExtentions, const uint32_t& count) {
            std::vector<vk::ExtensionProperties> extensionProperties = context.enumerateInstanceExtensionProperties();

            for (uint32_t i = 0; i < count; ++i) {
                const bool notSupported = std::ranges::none_of( 
                    extensionProperties, 
                    [glfwExtension = requiredExtentions[i]](const vk::ExtensionProperties& extensionProperty) {
                        return strcmp(extensionProperty.extensionName, glfwExtension) == 0;
                    }
                );

                if (notSupported) {
                    throw std::runtime_error("Required extension not supported: " + std::string(requiredExtentions[i]));
                }
            }
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

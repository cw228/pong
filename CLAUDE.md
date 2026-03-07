# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

General Vulkan, C++, and graphics programming notes are in `NOTES.md` — keep project-specific details here and general knowledge there.

## Tool Usage

**Context7 MCP:** Always use Context7 (`resolve-library-id` then `query-docs`) to look up library/API documentation when answering questions about APIs, generating code that uses external libraries, or providing setup/configuration steps. Do not rely solely on training data for library-specific details.

## Build Commands

```bash
cmake -B build/debug -G Ninja                                  # Configure debug
cmake -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release     # Configure release
cmake --build build/debug                                      # Build debug
cmake --build build/release                                    # Build release
./build/debug/pong                                             # Run debug
./build/release/pong                                           # Run release
```

The project requires `CMAKE_CXX_COMPILER` to be set to clang++ (usually via environment variable).

**macOS with Homebrew LLVM:** Set the `CMAKE_OSX_SYSROOT` environment variable (e.g., `export CMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path)`).

## Architecture

Vulkan application using **C++23**, **Vulkan-Hpp RAII wrappers** (`vk::raii::*`), and **GLFW** for windowing.

- **`src/main.cpp`** — calls `loadGameState()`, constructs `Window`, `InputState`, and `Renderer`, runs the main loop; loop order: `glfwPollEvents()` → `updateInputState()` → `updateGameState()` → `drawFrame()`
- **`src/inputstate.h`** / **`src/inputstate.cpp`** — `InputState` struct (mouse position, left mouse button, key states); `updateInputState()` polls GLFW each frame
- **`src/renderer.cpp`** / **`src/renderer.h`** — `Renderer` class encapsulates all Vulkan setup; takes `Window&` and `GameState&` (borrows, does not own); destructor calls `device.waitIdle()` before RAII members are destroyed
- **`src/window.h`** — `Window` struct wrapping `GLFWwindow*`; constructor takes `(uint32_t width, uint32_t height)`; destructor calls `glfwDestroyWindow` + `glfwTerminate`
- **`src/gamestate.h`** / **`src/gamestate.cpp`** — `GameState`, `Instance`, `Model`, `Texture`, `Text`, `HitBox` structs; `loadGameState()` constructs the game state in C++ (no JSON); `updateGameState(gameState, inputState, deltaTime)` runs game logic each frame

**`Window` is outside the try block in `main.cpp`** intentionally — it must outlive `Renderer` which may throw during construction.

**`Renderer` constructor flow:** calls `glfwSetWindowUserPointer` + `glfwSetFramebufferSizeCallback`, then `loadModels(gameState)` (loads all OBJ models into shared vertex/index buffers, builds `RenderModel` map), then `initVulkan()`:
`createInstance()` → `setupDebugMessenger()` → `createSurface()` → `pickPhysicalDevice()` → `findQueueFamilies()` → `createLogicalDevice()` → `getQueues()` → `createSwapchain()` → `createSwapchainImageViews()` → `createDescriptorSetLayout()` → `createGraphicsPipeline()` → `createCommandPool()` → `createColorResources()` → `createDepthResources()` → `createTextureImage()` → `createTextureImageView()` → `createTextureSampler()` → `createVertexBuffer()` → `createIndexBuffer()` → `createUniformBuffers()` → `createStorageBuffers()` → `createDescriptorPool()` → `createDescriptorSets()` → `createCommandBuffers()` → `createSyncObjects()`

## Game Data

Game state is constructed entirely in C++ in `loadGameState()` (`src/gamestate.cpp`). Models are OBJ files in `models/` (paddle, ball, barrier) and `models/font/` (one OBJ per character). No JSON data files or Python level builder are used at runtime.

## Shaders

Shaders are written in **Slang** (`shaders/shader.slang`) and compiled to SPIR-V by CMake using `slangc` from the Vulkan SDK. Output goes to `shaders/` (source directory, not build directory).

**Slang/SPIR-V note:** `SV_InstanceID` compiles to `gl_InstanceIndex - gl_BaseInstance` (always 0-based). To include the `firstInstance` offset, use `SV_StartInstanceLocation + SV_InstanceID`.

## Synchronization (Frames in Flight)

Uses `MAX_FRAMES_IN_FLIGHT = 2` with the following sync objects:

| Resource | Count | Indexed by | Protected by |
|----------|-------|------------|--------------|
| `presentCompleteSemaphores` | 2 (frames) | `frameIndex` | fence wait |
| `renderCompleteSemaphores` | N (images) | `imageIndex` | image re-acquisition |
| `frameCommandBuffers` | 2 (frames) | `frameIndex` | fence wait |
| `drawFences` | 2 (frames) | `frameIndex` | (the protection itself) |

`renderCompleteSemaphores` are per-image (not per-frame) because `presentKHR` runs asynchronously after the fence — the only guarantee an image's presentation is complete is when `acquireNextImage` returns that same image again.

Depth and MSAA color images are single-buffered (transient within each frame): `eUndefined` source layout + `loadOp::eClear` + `storeOp::eDontCare`.

## Key Configuration

- `VULKAN_HPP_NO_STRUCT_CONSTRUCTORS` — enables designated initializers for Vulkan structs
- `GLFW_INCLUDE_VULKAN` — GLFW includes Vulkan headers
- Validation layers enabled in debug builds (`#ifndef NDEBUG`)
- Precompiled headers for `vulkan_raii.hpp`, `GLFW/glfw3.h`, `glm/glm.hpp`
- MoltenVK is Vulkan 1.2 — Vulkan 1.3 features need extensions explicitly enabled on macOS (e.g., `VK_KHR_synchronization2`, `VK_KHR_dynamic_rendering`, `VK_KHR_portability_subset`)

## Troubleshooting

**Stale precompiled header after system update:** If you see an error about a header being modified since the PCH was built, delete the build directory and reconfigure.

**GPU hang / system freeze:** Common causes:
- Shader accesses a descriptor (UBO, texture) that was never bound.
- In `updateRenderState`, `renderState.instances` **must be cleared at the start** before repopulating from `GameState`. Without the clear, it grows unboundedly; once it exceeds `MAX_INSTANCES`, `memcpy` in `updateStorageBuffer` overflows the mapped storage buffer.

## Font / Text Assets

`models/font/` contains individual OBJ files (one per character: A-Z, 0-9, punctuation, 2D with `z=0`). `convert_font.py` generates these from `objects.json`. Characters are rendered as regular geometry.

## Dependencies

- **stb_image** / **tinyobjloader** — header-only; `#define ..._IMPLEMENTATION` in `src/libs.cpp`

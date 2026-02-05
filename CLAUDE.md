# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tool Usage

**Context7 MCP:** Always use Context7 (`resolve-library-id` then `query-docs`) to look up library/API documentation when answering questions about APIs, generating code that uses external libraries, or providing setup/configuration steps. Do not rely solely on training data for library-specific details.

## Build Commands

```bash
cmake -B build -G Ninja                    # Configure (debug by default)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release  # Configure release
cmake --build build                        # Build
./build/pong                               # Run
```

The project requires `CMAKE_CXX_COMPILER` to be set to clang++ (usually via environment variable).

**macOS with Homebrew LLVM:** When using Ninja, you must pass `CMAKE_OSX_SYSROOT` explicitly on the command line (environment variable alone doesn't work due to `clang-scan-deps` not receiving it):
```bash
cmake -B build -G Ninja -DCMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path)
```

## Architecture

Single-file Vulkan application in `src/main.cpp` using:
- **C++23** with designated initializers for Vulkan structs
- **Vulkan-Hpp RAII wrappers** (`vk::raii::*`) for automatic resource cleanup
- **GLFW** for window management and Vulkan surface creation

The `Pong` class encapsulates all Vulkan setup with initialization flow:
`initWindow()` → `createInstance()` → `setupDebugMessenger()` → `createSurface()` → `pickPhysicalDevice()` → `findQueueFamilies()` → `createLogicalDevice()` → `getQueues()` → `createSwapchain()` → `createImageViews()` → `createDescriptorSetLayout()` → `createGraphicsPipeline()` → `createCommandPool()` → `createVertexBuffer()` → `createIndexBuffer()` → `createUniformBuffers()` → `createCommandBuffers()` → `createSyncObjects()`

## Shaders

Shaders are written in **Slang** (`shaders/shader.slang`) and compiled to SPIR-V by CMake using `slangc` from the Vulkan SDK. Output goes to `build/shaders/`.

## Synchronization (Frames in Flight)

Uses `MAX_FRAMES_IN_FLIGHT = 2` with the following sync objects:

| Resource | Count | Indexed by | Protected by |
|----------|-------|------------|--------------|
| `presentCompleteSemaphores` | 2 (frames) | `frameIndex` | fence wait |
| `renderFinishedSemaphores` | N (images) | `imageIndex` | image re-acquisition |
| `commandBuffers` | 2 (frames) | `frameIndex` | fence wait |
| `drawFences` | 2 (frames) | `frameIndex` | (the protection itself) |

**Why the difference?** Fences tell us GPU work is done, but `presentKHR` runs asynchronously after that. The only guarantee that an image's presentation is complete is when `acquireNextImage` returns that same image again. So `renderFinishedSemaphores` must be tied to image index, not frame index.

**Semaphore reuse rules:** Binary semaphores must be used in strict signal/wait pairs. A semaphore can only be reused after both operations complete. The fence tells us `submit()` finished (so `presentCompleteSemaphore`'s wait completed → safe to reuse). But for `renderFinishedSemaphore`, the wait is in `presentKHR` which has no fence — we rely on `acquireNextImage` returning the same image to know that present completed.

**Input latency:** More frames in flight = more latency between input and display. With 2 frames in flight at 60Hz, expect ~50ms pipeline latency. This is why competitive games minimize frames in flight.

**Synchronization primitives summary:**

| Primitive | Synchronizes between |
|-----------|---------------------|
| Pipeline barrier | Commands within the same queue (not just within a single command buffer) |
| Semaphore | Queue submissions / queue operations (e.g., acquire → submit → present) |
| Fence | GPU and CPU |

**Access flags — prefer narrow over broad:** Use `eShaderSampledRead` instead of `eShaderRead` when the image is only used as a sampled texture. `eShaderRead` is a catch-all covering sampled reads, storage reads, and binding table reads — overly broad flags may cause unnecessary cache flushes.

**Init-time GPU uploads:** `device.waitIdle()` in `endSingleTimeCommands()` is fine for init-time staging uploads (textures, vertex/index buffers). For runtime uploads, replace with a fence on the specific submit.

## Wayland/Hyprland Notes

- Window won't appear until content is rendered (unlike X11)
- `GLFW_RESIZABLE = FALSE` causes issues on tiling compositors - avoid it
- Swapchain doesn't report `eErrorOutOfDateKHR` on resize - compositor scales the output instead. Must manually detect size changes via `glfwGetFramebufferSize()` or framebuffer callback

## Key Configuration

- `VULKAN_HPP_NO_STRUCT_CONSTRUCTORS` - Enables C++20 designated initializers for Vulkan structs
- `GLFW_INCLUDE_VULKAN` - GLFW includes Vulkan headers
- Validation layers enabled in debug builds (`#ifndef NDEBUG`)
- Precompiled headers for `vulkan_raii.hpp` and `GLFW/glfw3.h` to reduce compile times
- `VK_KHR_portability_subset` device extension required on macOS (MoltenVK)
- MoltenVK is Vulkan 1.2, so Vulkan 1.3 features need their extensions explicitly enabled on macOS (e.g., `VK_KHR_synchronization2`, `VK_KHR_dynamic_rendering`)

## C++ Conventions

**Parameter passing:**
- Vulkan handles (`vk::Image`, `vk::Buffer`) — pass by value (they're just 64-bit integers)
- RAII wrappers (`vk::raii::Image`, `vk::raii::CommandBuffer`) — pass by reference (non-copyable)
- Large structs, `std::string`, `std::vector` — pass by `const T&`

**Swapchain images:** `swapchain.getImages()` returns `vk::Image` (not `vk::raii::Image`) because the swapchain owns these images. You don't destroy them — the swapchain does when it's destroyed.

**Return values:** Returning RAII objects by value is idiomatic. The compiler uses move semantics or copy elision (constructs directly in caller's stack frame). No explicit `std::move` needed on return.

**RAII wrapper initialization:** Use `= nullptr` for empty handles to be assigned later (clearer than `({})`). RAII wrappers implicitly convert to bare handles, or use `*wrapper` to explicitly extract the handle.

**RAII wrappers and containers:** RAII wrappers are non-copyable. When adding to a `std::vector`, either use `std::move(obj)` with `push_back`, or pass the factory function return value directly (it's already an rvalue): `vec.push_back(createThing(...))`.

**Descriptor pool sizing:** `maxSets` and `pPoolSizes` are independent limits — `maxSets` caps total descriptor sets allocated, `pPoolSizes` caps total descriptors per type. Both must be satisfied for allocation to succeed.

## Shader Notes

**MVP matrix order:** In Slang/HLSL, transformations apply right-to-left. Use `mul(projection, mul(view, mul(model, position)))` to get the correct `projection * view * model * position` order.

## Troubleshooting

**Stale precompiled header after system update:** If you see an error about a header being modified since the PCH was built, delete the PCH and rebuild:
```bash
rm -f build/CMakeFiles/pong.dir/cmake_pch.hxx.pch && cmake --build build
```

**GPU hang / system freeze:** If running the app freezes the system (black screen, blinking cursor), the GPU is hanging. Common cause: shader accesses a descriptor (UBO, texture) that was never bound. Always ensure descriptor sets are created, updated, and bound before draw calls that use them.

## Dependencies

- **stb_image** (`pacman -S stb`) — header-only image loading. Requires `#define STB_IMAGE_IMPLEMENTATION` before include in exactly one source file. `stbi_uc` is a typedef for `unsigned char`.

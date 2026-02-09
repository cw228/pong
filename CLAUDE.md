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

**Barrier source/destination masks:** A barrier has four synchronization components:
- Source stage: what must finish *executing*
- Source access: what writes must be *flushed from cache* (availability — data leaves the writer's private cache into shared memory)
- Dest stage: what must *wait* before executing
- Dest access: what caches must be *invalidated* so reads see the flushed data (visibility — reader's stale cache entries are discarded)

Availability (flush) is the expensive part and only needs to happen once. Visibility (invalidation) is cheap and done per-consumer. Access flags map to hardware caches (e.g., `eColorAttachmentWrite` → ROP cache, `eShaderSampledRead` → texture unit cache), not to pipeline stages — that's why you specify both stage and access type.

**Image layout transitions and content preservation:** Any source layout other than `eUndefined` implicitly means "preserve the contents" — the driver uses the source layout to know how the data is currently arranged so it can convert it. `eUndefined` means "I'm not telling you the current layout," so the driver *can't* preserve the data and may discard it. Specifying the wrong source layout (not matching the image's actual current layout) is undefined behavior — the driver trusts you, and validation layers track this.

**Depth image uses a single buffer** (not one per frame in flight) because the pipeline barrier before each frame's depth writes synchronizes against prior depth writes on the same queue. This serializes the depth-related work between frames while allowing other stages (vertex processing, color writes) to overlap. Combined with `eUndefined` source layout + `loadOp::eClear` + `storeOp::eDontCare`, the depth buffer is purely transient within each frame.

**Image initial layouts:** Images are created with `initialLayout` which can only be `eUndefined` or `ePrelinear`. Almost always use `eUndefined` — `ePrelinear` is only for CPU-mapped linearly-tiled images. The first layout transition for most images will therefore have `eUndefined` as the source.

**Semaphores carry implicit memory dependencies:** A semaphore signal makes all available memory visible after the corresponding wait. This is why the present-transition barrier can have empty destination access/stage — the barrier flushes color writes (availability), and the `renderFinishedSemaphore` between submit and `presentKHR` handles visibility to the presentation engine. Chain: barrier flushes → semaphore signal picks up available data → semaphore wait makes it visible.

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

**Vulkan-Hpp struct pointer semantics:** Structs like `vk::DependencyInfo`, `vk::RenderingInfo`, etc. store raw pointers (`pImageMemoryBarriers`, etc.), not copies. Mutating the pointed-to data and resubmitting the parent struct works without recreating it — useful for reusing a barrier in a loop (e.g., mipmap generation).

**Descriptor pool sizing:** `maxSets` and `pPoolSizes` are independent limits — `maxSets` caps total descriptor sets allocated, `pPoolSizes` caps total descriptors per type. Both must be satisfied for allocation to succeed.

## Multisampling (MSAA)

**Depth must be multisampled at the same rate as color.** Each color sample needs its own depth value for an independent depth test. With single-sampled depth, the depth test is per-pixel (pass/fail for all samples), so draw order affects correctness — a closer triangle drawn first would cause a farther triangle to be entirely discarded, leaving uncovered samples as background instead of showing the farther triangle. Per-sample depth eliminates draw-order dependence.

**`getMaxUsableSampleCount`:** `vk::SampleCountFlags` is a bitmask where bit N = 2^N samples. Casting to `uint32_t` and using `std::bit_floor` (C++20, `<bit>`) finds the highest supported power-of-2 sample count. Intersect `framebufferColorSampleCounts & framebufferDepthSampleCounts` to ensure both attachments support the chosen count.

**Depth resolve:** MSAA resolve only operates on color — it averages color samples into the single-sample framebuffer. Per-sample depth values are discarded (hence `storeOp::eDontCare` on the depth attachment). Averaging depth values would be meaningless.

## Shader Notes

**MVP matrix order:** In Slang/HLSL, transformations apply right-to-left. Use `mul(projection, mul(view, mul(model, position)))` to get the correct `projection * view * model * position` order.

**LOD (Level of Detail):** LOD 0 = mip level 0 = full-resolution base image. Higher LOD selects higher mip levels (smaller, less detailed). `minLod = 0.0f` and `maxLod = LodClampNone` allows the sampler to use the full mip range.

**Depth values:** The vertex shader outputs clip-space `z` via `SV_Position`. Fixed-function hardware then does perspective division (`z / w`) to get NDC depth [0, 1] (Vulkan range, unlike OpenGL's [-1, 1]), then the viewport transform maps it to framebuffer depth: `depth = minDepth + ndc.z * (maxDepth - minDepth)`. No shader code needed for depth — the projection matrix encodes the z mapping.

## Troubleshooting

**Stale precompiled header after system update:** If you see an error about a header being modified since the PCH was built, delete the PCH and rebuild:
```bash
rm -f build/CMakeFiles/pong.dir/cmake_pch.hxx.pch && cmake --build build
```

**GPU hang / system freeze:** If running the app freezes the system (black screen, blinking cursor), the GPU is hanging. Common cause: shader accesses a descriptor (UBO, texture) that was never bound. Always ensure descriptor sets are created, updated, and bound before draw calls that use them.

## Dependencies

- **stb_image** (`pacman -S stb`) — header-only image loading. Requires `#define STB_IMAGE_IMPLEMENTATION` before include in exactly one source file. `stbi_uc` is a typedef for `unsigned char`.

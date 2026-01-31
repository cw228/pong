# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
`initWindow()` → `createInstance()` → `setupDebugMessenger()` → `createSurface()` → `pickPhysicalDevice()` → `findQueueFamilies()` → `createLogicalDevice()` → `createSwapchain()` → `createImageViews()`

## Key Configuration

- `VULKAN_HPP_NO_STRUCT_CONSTRUCTORS` - Enables C++20 designated initializers for Vulkan structs
- `GLFW_INCLUDE_VULKAN` - GLFW includes Vulkan headers
- Validation layers enabled in debug builds (`#ifndef NDEBUG`)
- Precompiled headers for `vulkan_raii.hpp` and `GLFW/glfw3.h` to reduce compile times
- `VK_KHR_portability_subset` device extension required on macOS (MoltenVK)

## Troubleshooting

**Stale precompiled header after system update:** If you see an error about a header being modified since the PCH was built, delete the PCH and rebuild:
```bash
rm -f build/CMakeFiles/pong.dir/cmake_pch.hxx.pch && cmake --build build
```

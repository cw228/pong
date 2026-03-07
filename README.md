# Pong

A Pong game built with C++23 and Vulkan, using Vulkan-Hpp RAII wrappers and GLFW for windowing.

## Dependencies

- **C++ compiler** with C++23 support (clang++ recommended)
- **CMake** 3.28+
- **Ninja** build system
- **Vulkan SDK** 1.3.296.0+ (includes `slangc` shader compiler)
- **GLFW** — windowing and input
- **GLM** — math library
- **stb_image** — image loading (header-only)
- **tinyobjloader** — OBJ model loading (header-only)

The Vulkan SDK must be installed separately from [LunarG](https://vulkan.lunarg.com/sdk/home) on macOS and Windows. On Arch Linux it is available as a package.

### Arch Linux

```bash
sudo pacman -S clang cmake ninja glfw-x11 glm vulkan-devel stb tinyobjloader
```

Use `glfw-wayland` instead of `glfw-x11` if you are running Wayland.

### macOS (Homebrew)

```bash
brew install llvm cmake ninja glfw glm stb tinyobjloader
```

### Windows

Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) and use [vcpkg](https://vcpkg.io/) for the remaining libraries:

```powershell
vcpkg install glfw3 glm stb tinyobjloader
```

Then pass the vcpkg toolchain file when configuring CMake:

```powershell
cmake -B build/debug -G Ninja -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
```

## Environment Setup

Set `CMAKE_CXX_COMPILER` to clang++. On macOS with Homebrew LLVM, also set the sysroot:

```bash
export CMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++
export CMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path)  # macOS only
```

Make sure `VULKAN_SDK` is set (the Vulkan SDK setup script typically handles this).

## Build and Run

```bash
cmake -B build/debug -G Ninja                                  # Configure (debug)
cmake --build build/debug                                      # Build
./build/debug/pong                                             # Run
```

For a release build:

```bash
cmake -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
./build/release/pong
```

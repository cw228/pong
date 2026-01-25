cmake_minimum_required(VERSION 3.28)
project(pong LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

# Create executable
add_executable(pong src/main.cpp)

# Link libraries
target_link_libraries(pong PRIVATE
    Vulkan::Vulkan
    glfw
)


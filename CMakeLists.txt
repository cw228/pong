cmake_minimum_required(VERSION 3.28)
project(pong LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Find required packages
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

# stb is header-only, just verify it's installed
find_path(STB_INCLUDE_DIR stb/stb_image.h REQUIRED)

# tinyobjloader is header-only, just verify it's installed
find_path(TINYOBJLOADER_INCLUDE_DIR tiny_obj_loader.h REQUIRED)

# Create executable
add_executable(pong src/main.cpp src/libs.cpp)

# Link libraries
target_link_libraries(pong PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
)

target_include_directories(pong PRIVATE ${STB_INCLUDE_DIR} ${TINYOBJLOADER_INCLUDE_DIR})

# Vulkan-Hpp config (enables designated initializers)
target_compile_definitions(pong PRIVATE
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
    GLFW_INCLUDE_VULKAN
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_ENABLE_EXPERIMENTAL
)

# Precompile heavy headers
target_precompile_headers(pong PRIVATE
    <vulkan/vulkan_raii.hpp>
    <GLFW/glfw3.h>
    <glm/glm.hpp>
    <glm/gtc/matrix_transform.hpp>
    <glm/gtx/hash.hpp>
)

# Find slangc compiler (included in Vulkan SDK 1.3.296.0+)
find_program(slangc_exe slangc HINTS $ENV{VULKAN_SDK}/bin)
if(NOT slangc_exe)
    message(FATAL_ERROR "slangc not found. Make sure Vulkan SDK is installed and VULKAN_SDK is set.")
endif()

# Function to compile slang shaders to SPIR-V
function(compile_slang_shader source_file entry_point)
    cmake_path(GET source_file STEM file_stem)
    set(output_file "${CMAKE_BINARY_DIR}/shaders/${file_stem}_${entry_point}.spv")
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
        COMMAND ${slangc_exe}
            ${source_file}
            -target spirv
            -profile spirv_1_4
            -emit-spirv-directly
            -fvk-use-entrypoint-name
            -entry ${entry_point}
            -o ${output_file}
        DEPENDS ${source_file}
        COMMENT "Compiling ${file_stem}.slang (${entry_point})"
    )
    set(shader_output ${output_file})
    return(PROPAGATE shader_output)
endfunction()

# Compile shaders
compile_slang_shader(${CMAKE_SOURCE_DIR}/shaders/shader.slang vertMain)
list(APPEND shader_outputs ${shader_output})
compile_slang_shader(${CMAKE_SOURCE_DIR}/shaders/shader.slang fragMain)
list(APPEND shader_outputs ${shader_output})

# Custom target for all shaders
add_custom_target(shaders DEPENDS ${shader_outputs})
add_dependencies(pong shaders)


# NOTES.md

General Vulkan, C++, and graphics programming notes accumulated while building this project.

## Vulkan Synchronization

**Synchronization primitives:**

| Primitive | Synchronizes between |
|-----------|---------------------|
| Pipeline barrier | Commands within the same queue (not just within a single command buffer) |
| Semaphore | Queue submissions / queue operations (e.g., acquire → submit → present) |
| Fence | GPU and CPU |

**Barrier source/destination masks:** A barrier has four synchronization components:
- Source stage: what must finish *executing*
- Source access: what writes must be *flushed from cache* (availability — data leaves the writer's private cache into shared memory)
- Dest stage: what must *wait* before executing
- Dest access: what caches must be *invalidated* so reads see the flushed data (visibility — reader's stale cache entries are discarded)

Availability (flush) is the expensive part and only needs to happen once. Visibility (invalidation) is cheap and done per-consumer. Access flags map to hardware caches (e.g., `eColorAttachmentWrite` → ROP cache, `eShaderSampledRead` → texture unit cache), not to pipeline stages — that's why you specify both stage and access type.

**Barrier `srcAccessMask` when source layout is `eUndefined`:** Use `{}` (empty). Since `eUndefined` means "discard contents," there are no prior writes to flush. A non-empty `srcAccessMask` won't crash but is semantically wrong and may produce validation warnings.

**Image layout transitions and content preservation:** Any source layout other than `eUndefined` implicitly means "preserve the contents" — the driver uses the source layout to know how the data is currently arranged so it can convert it. `eUndefined` means "I'm not telling you the current layout," so the driver *can't* preserve the data and may discard it. Specifying the wrong source layout (not matching the image's actual current layout) is undefined behavior — the driver trusts you, and validation layers track this.

**Image initial layouts:** Images are created with `initialLayout` which can only be `eUndefined` or `ePrelinear`. Almost always use `eUndefined` — `ePrelinear` is only for CPU-mapped linearly-tiled images.

**Semaphores carry implicit memory dependencies:** A semaphore signal makes all available memory visible after the corresponding wait. This is why a present-transition barrier can have empty destination access/stage — the barrier flushes color writes (availability), and the semaphore between submit and `presentKHR` handles visibility to the presentation engine.

**Semaphore reuse rules:** Binary semaphores must be used in strict signal/wait pairs. A semaphore can only be reused after both operations complete.

**Access flags — prefer narrow over broad:** Use `eShaderSampledRead` instead of `eShaderRead` when the image is only used as a sampled texture. `eShaderRead` is a catch-all covering sampled reads, storage reads, and binding table reads — overly broad flags may cause unnecessary cache flushes.

**Init-time GPU uploads:** `device.waitIdle()` in `endSingleTimeCommands()` is fine for init-time staging uploads (textures, vertex/index buffers). For runtime uploads, replace with a fence on the specific submit.

**Input latency:** More frames in flight = more latency between input and display. With 2 frames in flight at 60Hz, expect ~50ms pipeline latency.

## Multisampling (MSAA)

**Depth must be multisampled at the same rate as color.** Each color sample needs its own depth value for an independent depth test. With single-sampled depth, the depth test is per-pixel (pass/fail for all samples), so draw order affects correctness — a closer triangle drawn first would cause a farther triangle to be entirely discarded, leaving uncovered samples as background instead of showing the farther triangle. Per-sample depth eliminates draw-order dependence.

**`getMaxSampleCount`:** `vk::SampleCountFlags` is a bitmask where bit N = 2^N samples. Casting to `uint32_t` and using `std::bit_floor` (C++20, `<bit>`) finds the highest supported power-of-2 sample count. Intersect `framebufferColorSampleCounts & framebufferDepthSampleCounts` to ensure both attachments support the chosen count.

**Depth resolve:** MSAA resolve only operates on color — it averages color samples into the single-sample framebuffer. Per-sample depth values are discarded (hence `storeOp::eDontCare` on the depth attachment). Averaging depth values would be meaningless.

**Dynamic rendering MSAA resolve:** With dynamic rendering, resolve is configured inline on the primary (MSAA) color attachment via `resolveMode`, `resolveImageView`, and `resolveImageLayout` — not by adding a second color attachment. (With legacy `VkRenderPass`, resolve was a separate `pResolveAttachments` array in the subpass.) The `loadOp`/`storeOp` on the attachment apply only to the MSAA image; the resolved result is always stored to `resolveImageView` implicitly. Use `storeOp::eDontCare` on the MSAA image since its contents aren't needed after resolve.

**Resolve synchronization:** The resolve operation uses `eColorAttachmentOutput` stage and `eColorAttachmentWrite` access — same as regular color writes. There is no separate resolve stage in Vulkan's synchronization model.

## Shader Notes

**MVP matrix order:** In Slang/HLSL, transformations apply right-to-left. Use `mul(projection, mul(view, mul(model, position)))` to get the correct `projection * view * model * position` order.

**GLM model matrix (TRS):** To get net matrix T×R×S (scale applied first to vertices), call: `glm::translate(identity, pos)`, then `glm::rotate(...)`, then `glm::scale(...)`. GLM **right-multiplies** each transform — `glm::translate(m, v)` returns `m * T`, so calling translate→rotate→scale chains as `T * R * S`. Getting the order wrong causes positions to be affected by scale (e.g. with S*R*T, the position vector itself gets scaled). Rotation is around the Z-axis for 2D; `rotation` field is in degrees, convert with `glm::radians()`.

**LOD (Level of Detail):** LOD 0 = mip level 0 = full-resolution base image. Higher LOD selects higher mip levels (smaller, less detailed). `minLod = 0.0f` and `maxLod = LodClampNone` allows the sampler to use the full mip range.

**Depth values:** The vertex shader outputs clip-space `z` via `SV_Position`. Fixed-function hardware then does perspective division (`z / w`) to get NDC depth [0, 1] (Vulkan range, unlike OpenGL's [-1, 1]), then the viewport transform maps it to framebuffer depth: `depth = minDepth + ndc.z * (maxDepth - minDepth)`. No shader code needed for depth — the projection matrix encodes the z mapping.

## C++ / Vulkan-Hpp Conventions

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

**`std::unordered_map::operator[]`** default-constructs the value if the key is absent, so an explicit `contains()` check before insertion is always redundant.

**Destruction order:** Declare `Window window` before `Renderer renderer` so that `renderer` is destroyed first (reverse construction order). The destructor body runs before member destructors, so `device.waitIdle()` in the destructor ensures the device is idle before RAII members are cleaned up.

## Wayland/Hyprland Notes

- Window won't appear until content is rendered (unlike X11)
- `GLFW_RESIZABLE = FALSE` causes issues on tiling compositors — avoid it
- Swapchain doesn't report `eErrorOutOfDateKHR` on resize — compositor scales the output instead. Must manually detect size changes via `glfwGetFramebufferSize()` or framebuffer callback

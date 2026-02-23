# Storage Buffer for Model Matrices

Replace the single `model` matrix in the UBO with an SSBO holding all instance
matrices, indexed in the vertex shader via `SV_InstanceID`. One `drawIndexed`
call per entity with `instanceCount` set to the number of instances and
`firstInstance` pointing to that entity's base offset in the SSBO.

---

## 1. `renderer.h` — data structure changes

- Add `int32_t firstInstance` to `RenderEntity` (base offset into the SSBO for
  this entity's instances).
- Add `int32_t totalInstances` to `RenderState` (total across all entities/levels;
  determines the SSBO size).
- Remove `model` from `UniformBufferObject`; keep only `view` and `projection`.
- Add SSBO member vectors (parallel to the existing uniform buffer vectors):
  ```
  std::vector<vk::raii::Buffer>       storageBuffers;
  std::vector<vk::raii::DeviceMemory> storageBuffersMemory;
  std::vector<void*>                  storageBuffersMapped;
  ```
- Add `void createStorageBuffers()` declaration.

---

## 2. `createRenderState()` — assign `firstInstance` and count total instances

After populating `renderState.entityInstances`, make a second pass to assign each
`RenderEntity` its `firstInstance` offset:

```
int32_t offset = 0;
for (auto& [entityId, renderEntity] : renderState.entities) {
    renderEntity.firstInstance = offset;
    offset += renderState.entityInstances[entityId].size();
}
renderState.totalInstances = offset;
```

The matrices are already stored in `renderState.entityInstances[entityId][instanceId]`
in insertion order. The SSBO will be filled by iterating entities in the same order
as this pass so that `firstInstance + localIndex` correctly addresses each matrix.

---

## 3. `createStorageBuffers()` — new method

Mirror `createUniformBuffers()` exactly, but:
- Size: `sizeof(glm::mat4) * renderState.totalInstances`
- Usage flag: `eStorageBuffer` (no staging needed since it's host-visible)
- Memory: `eHostVisible | eHostCoherent`, persistently mapped

Call it from `initVulkan()` immediately after `createUniformBuffers()`.

---

## 4. `createDescriptorSetLayout()` — add binding 2

Add a third entry to the `bindings` array:
```cpp
vk::DescriptorSetLayoutBinding{
    .binding        = 2,
    .descriptorType = vk::DescriptorType::eStorageBuffer,
    .descriptorCount = 1,
    .stageFlags     = vk::ShaderStageFlagBits::eVertex
}
```

---

## 5. `createDescriptorPool()` — add SSBO pool size

Add to the `poolSizes` array:
```cpp
vk::DescriptorPoolSize{
    .type            = vk::DescriptorType::eStorageBuffer,
    .descriptorCount = MAX_FRAMES_IN_FLIGHT,
}
```

---

## 6. `createDescriptorSets()` — write binding 2

In the per-frame loop, add a `vk::DescriptorBufferInfo` pointing to
`storageBuffers[i]` with range `sizeof(glm::mat4) * renderState.totalInstances`,
and a corresponding `vk::WriteDescriptorSet` at `dstBinding = 2` with type
`eStorageBuffer`.

---

## 7. `updateUniformBuffer()` — remove model, write SSBO

- Remove the `ubo.model` line (the field no longer exists in `UniformBufferObject`).
- Write model matrices into `storageBuffersMapped[currentFrameIndex]`. Iterate
  `renderState.entities` in the same order used when assigning `firstInstance`,
  and for each entity iterate its instances:
  ```
  for each entity (same order as step 2):
      for each instance in renderState.entityInstances[entityId]:
          write instance.modelMatrix at (firstInstance + localIndex) * sizeof(mat4)
  ```
  Use `memcpy` with the appropriate byte offset into the mapped pointer, same
  pattern as the UBO write.

---

## 8. `recordFrameCommandBuffer()` — per-entity instanced draws

Replace the single `drawIndexed` call (currently line 257) with a loop:

```cpp
for (auto& [entityId, renderEntity] : renderState.entities) {
    uint32_t instanceCount = renderState.entityInstances[entityId].size();
    commandBuffer.drawIndexed(
        renderEntity.indexCount,
        instanceCount,
        renderEntity.indexOffset,
        renderEntity.vertexOffset,
        renderEntity.firstInstance   // SV_InstanceID starts here
    );
}
```

---

## 9. Shader (`shaders/shader.slang`)

- Remove `model` from the `UniformBuffer` struct; keep `view` and `projection`.
- Add an SSBO at binding 2:
  ```slang
  StructuredBuffer<float4x4> modelMatrices;
  ```
- Add `SV_InstanceID` as a parameter to `vertMain`:
  ```slang
  VertexOutput vertMain(VertexInput input, uint instanceIndex : SV_InstanceID)
  ```
- Use it to look up the model matrix:
  ```slang
  float4x4 model = modelMatrices[instanceIndex];
  output.position = mul(ubo.projection, mul(ubo.view, mul(model, float4(input.position, 1.0))));
  ```

> **Note:** In Slang targeting SPIR-V, `SV_InstanceID` maps to `gl_InstanceIndex`,
> which includes the `firstInstance` offset passed to `drawIndexed`. This is
> intentional — it means `instanceIndex` directly addresses the correct row in the
> SSBO without any additional arithmetic.

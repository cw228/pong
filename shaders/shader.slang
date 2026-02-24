struct VertexInput {
    float3 position;
    float3 color;
    float2 textureCoordinates;
};

struct UniformBuffer {
    float4x4 view;
    float4x4 projection;
};

[vk::binding(0, 0)] ConstantBuffer<UniformBuffer> ubo;

struct VertexOutput {
    float4 position : SV_Position;
    float3 color;
    float2 textureCoordinates; 
};

[vk::binding(2, 0)] StructuredBuffer<float4x4> modelMatrices;

[shader("vertex")]
VertexOutput vertMain(VertexInput input, int instanceId : SV_InstanceID, int firstInstance : SV_StartInstanceLocation) {
    VertexOutput output;
    float4x4 model = modelMatrices[firstInstance + instanceId];
    output.position = mul(ubo.projection, mul(ubo.view, mul(model, float4(input.position, 1.0))));
    output.color = input.color;
    output.textureCoordinates = input.textureCoordinates;
    return output;
}

[vk::binding(1, 0)] Sampler2D texture;

[shader("fragment")]
float4 fragMain(VertexOutput vertex) : SV_Target {
    return texture.Sample(vertex.textureCoordinates);
}

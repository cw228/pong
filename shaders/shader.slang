struct VertexInput {
    float2 position;
    float3 color;
};

struct VertexOutput {
    float4 position : SV_Position;
    float3 color;
};

[shader("vertex")]
VertexOutput vertMain(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput vertex) : SV_Target {
    float3 color = vertex.color;
    return float4(color, 1.0);
}

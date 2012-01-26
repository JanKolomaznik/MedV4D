
struct ViewSetup
{
	double4x4 modelViewProj;
	double4x4 modelMatrix;
	double4x4 projMatrix;
	double4x4 viewMatrix;
};

struct ImageData3D
{
	sampler3D data: TEX0;
	int3 size;
	float3 realSize;
	float3 realMinimum;
	float3 realMaximum;
};

struct TransferFunction1D
{
	sampler1D data: TEX1;
	float2 interval;
	int sampleCount;
};

struct FragmentOutput
{
	float4 color : COLOR;
};

struct VertexOutput
{
	float4 position : POSITION;
	float4 color    : COLOR0;
	float3 texCoord : TEXCOORD0;
};

struct Material
{
	float3	Ka;
	float3	Kd;
	float3	Ks;
	float	shininess;
};

struct Light
{
	float3 color;
	float3 ambient;
	float3 position;
};

struct BoundingBox
{
	float3	vertices[8];
};

struct Plane
{
	float3 point;
	float3 normal;
};
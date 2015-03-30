struct ImageData3D
{
	sampler3D data;
	ivec3 size;
	vec3 realSize;
	vec3 realMinimum;
	vec3 realMaximum;
	vec3 elementExtents;
};

struct TransferFunction1D
{
	sampler1D data;
	vec2 interval;
	int sampleCount;
};

struct TransferFunction2D
{
	sampler2D data;
	vec2 from;
	vec2 to;
	int xSamples;
	int ySamples;
	bool eigenvalueProcessPrimary;
	bool eigenvalueProcessSecondary;
	vec3 primaryEigenvalueParameters;
	int primaryValuesMultiplier;
	vec3 secondaryEigenvalueParameters;
	int secondaryValuesMultiplier;
};

struct Camera
{
	vec3 eyePosition;
	vec3 viewDirection;
	vec3 upDirection;	
};

struct ViewSetup
{
	mat4 modelViewProj;
	mat4 modelMatrix;
	mat4 projMatrix;
	mat4 viewMatrix;
};

struct Light
{
	vec3 color;
	vec3 ambient;
	vec3 position;
};

struct Material
{
	vec3	Ka;
	vec3	Kd;
	vec3	Ks;
	float	shininess;
};



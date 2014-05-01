
in vec3 positionInImage;
out vec4 fragmentColor;

uniform sampler2D gNoiseMap;
uniform float gJitterStrength = 1.0f;
uniform vec2 gNoiseMapSize;

uniform Camera gCamera;
uniform ViewSetup gViewSetup;
uniform vec2 gWindowSize;
uniform sampler2D gDepthBuffer;

uniform float gRenderingSliceThickness;

void main(void)
{
	vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
	vec3 coordinates = positionInImage;

	vec3 dir = normalize(coordinates - gCamera.eyePosition);
	float value = -1.0;

	StepInfo info = initInfo(coordinates);

	for (int i = 0; i < 1000; ++i) {
		vec3 point = coordinates + i * dir;

		vec4 depth_vec = gViewSetup.modelViewProj * vec4(point, 1.0);
		float currentDepth = (depth_vec.z / depth_vec.w + 1.0) * 0.5;
		float depth = texture(gDepthBuffer, gl_FragCoord.st / gWindowSize).x;
		if (depth < currentDepth) {
			break;
		}

		info = doStep(info, coordinates);
	}
	fragmentColor = colorFromStepInfo(info);
}

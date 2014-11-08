
in vec3 positionInImage;
out vec4 fragmentColor;

void main(void)
{
	vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
	vec3 coordinates = positionInImage;

#ifdef ENABLE_JITTERING
	float offset = texture(gNoiseMap, gl_FragCoord.xy / gNoiseMapSize.xy ).r * gJitterStrength * gRenderingSliceThickness;
	coordinates = coordinates + offset * gCamera.viewDirection;
#endif //ENABLE_JITTERING

	vec3 dir = normalize(coordinates - gCamera.eyePosition);
	float value = -1.0;

	StepInfo info = initInfo(coordinates);

	for (int i = 0; i < 1000; ++i) {
		vec3 point = coordinates + gRenderingSliceThickness * i * dir;

		vec4 depth_vec = gViewSetup.modelViewProj * vec4(point, 1.0);
		float currentDepth = (depth_vec.z / depth_vec.w + 1.0) * 0.5;
		float depth = texture(gDepthBuffer, gl_FragCoord.st / gWindowSize).x;
		if (depth < currentDepth) {
			break;
		}

#ifdef USE_MASK
		vec3 maskCoordinates = texCoordsFromPosition(point, gMaskData3D);
		if (texture(gMaskData3D.data, maskCoordinates).x < 0.5) {
			//info.color = vec4(1.0, 0.0, 0.0, 0.5);
			//break;
			continue;
		}	
#endif // USE_MASK

		info = doStep(info, point, dir);
	}
	fragmentColor = colorFromStepInfo(info);
}

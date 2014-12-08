
uniform sampler2D gNoiseMap;
uniform float gJitterStrength = 1.0f;
uniform vec2 gNoiseMapSize;

uniform Camera gCamera;
uniform ViewSetup gViewSetup;
uniform vec2 gWindowSize;
uniform sampler2D gDepthBuffer;

uniform float gRenderingSliceThickness;

uniform ImageData3D gPrimaryImageData3D;
uniform vec2 gMappedIntervalBands;

uniform ImageData3D gSecondaryImageData3D;
uniform vec2 gSecondaryMappedIntervalBands;

uniform ImageData3D gMaskData3D;

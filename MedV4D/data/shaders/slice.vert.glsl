
uniform ViewSetup gViewSetup;

in vec3 vertex;
out vec3 positionInImage;

void main(void) {
	// compute position
	gl_Position = gViewSetup.modelViewProj * vec4(vertex, 1.0);
	positionInImage = vertex;
}

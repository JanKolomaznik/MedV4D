 
#version 130

uniform mat4 modelViewProj; // model-view-projection matrix

in vec3 vertex;

out vec3 positionInImage;

void main(void) {
	// compute position
	gl_Position = modelViewProj * vec4(vertex, 1.0);
	positionInImage = vertex;
}
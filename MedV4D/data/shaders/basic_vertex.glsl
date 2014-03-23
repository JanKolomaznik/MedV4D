
#version 130

struct ViewSetup
{
	mat4 modelViewProj;
	mat4 modelMatrix;
	mat4 projMatrix;
	mat4 viewMatrix;
};

uniform ViewSetup gViewSetup;

//attribute vec3 vertex;
//attribute vec3 normal;
//attribute vec2 uv1;
//attribute vec4 tangent;

//uniform mat4 _mv; // model-view matrix
//uniform mat4 modelViewProj; // model-view-projection matrix
//uniform mat3 _norm; // normal matrix
//uniform float _time; // time in seconds

//varying vec2 uv;
//varying vec3 n;

in vec3 vertex;

void main(void) {
	// compute position
	gl_Position = gViewSetup.modelViewProj * vec4(vertex, 1.0); 
}
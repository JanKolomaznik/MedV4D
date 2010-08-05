#ifndef CG_SHADER_TOOLS_H
#define CG_SHADER_TOOLS_H

#ifdef USE_CG

#include "GUI/utils/OGLDrawing.h"
#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgGL.h>
#include <string>
#include "common/Common.h"
#include "GUI/utils/DrawingTools.h"
#include "GUI/utils/Camera.h"

class CgException: public M4D::ErrorHandling::ExceptionBase
{
public:
	CgException( std::string name ) throw() : ExceptionBase( name ) {}
	~CgException() throw(){}
};

struct AShaderConfig
{
	void
	Initialize(	CGcontext   				&cgContext,
			const boost::filesystem::path 		&fragmentProgramFile, 
			const std::string 			&fragmentProgramName 
			);
	void
	Finalize();

	void
	Enable();

	void
	Disable();

	CGcontext   cgContext;

	CGprofile   cgFragmentProfile;
	CGprogram   cgFragmentProgram;
};

struct CgBrightnessContrastShaderConfig : public AShaderConfig
{
	void
	Initialize(	CGcontext   				&cgContext,
			const boost::filesystem::path 		&fragmentProgramFile, 
			const std::string 			&fragmentProgramName 
			);

	void
	Enable();

	void
	Disable();
	
	GLuint			textureName;
	Vector< float, 2 >	brightnessContrast;

	CGparameter cgFragmentParam_Texture;
	CGparameter cgFragmentParam_BrightnessContrast;
};


struct CgMaskBlendShaderConfig : public AShaderConfig
{
	void
	Initialize(	CGcontext   				&cgContext,
			const boost::filesystem::path 		&fragmentProgramFile, 
			const std::string 			&fragmentProgramName 
			);

	void
	Enable();

	void
	Disable();

	GLuint			textureName;
	Vector< float, 4 >	blendColor;

	CGparameter 		cgFragmentParam_Texture;
	CGparameter 		cgFragmentParam_BlendColor;
};

struct CgSimpleTransferFunctionShaderConfig : public AShaderConfig
{
	void
	Initialize(	CGcontext   				&cgContext,
			const boost::filesystem::path 		&fragmentProgramFile, 
			const std::string 			&fragmentProgramName 
			);

	void
	Enable();

	void
	Disable();
	
	GLuint			dataTexture;
	GLuint			transferFunctionTexture;

	CGparameter cgFragmentParam_Data;
	CGparameter cgFragmentParam_TransferFunction;
};

struct CgTransferFunctionShadingShaderConfig : public AShaderConfig
{
	void
	Initialize(	CGcontext   				&cgContext,
			const boost::filesystem::path 		&fragmentProgramFile, 
			const std::string 			&fragmentProgramName 
			);

	void
	Enable();

	void
	Disable();
	
	GLuint			dataTexture;
	GLuint			transferFunctionTexture;
	Vector< float, 3 >	eyePosition;
	Vector< float, 3 >	lightPosition;
	Vector< float, 3 >	sliceNormal;
	float			sliceSpacing;

	CGparameter cgFragmentParam_Data;
	CGparameter cgFragmentParam_TransferFunction;
	CGparameter cgFragmentParam_EyePosition;
	CGparameter cgFragmentParam_LightPosition;
	CGparameter cgFragmentParam_SliceSpacing;
	CGparameter cgFragmentParam_SliceNormal;
};


void 
CheckForCgError( const std::string &situation, CGcontext &context );

void
GLDrawVolumeSlicesForVertexShader(
		M4D::BoundingBox3D	bbox,
		Camera		camera,
		unsigned 	numberOfSteps,
		float		cutPlane = 1.0f
		);


#endif /*USE_CG*/

#endif /*CG_SHADER_TOOLS_H*/

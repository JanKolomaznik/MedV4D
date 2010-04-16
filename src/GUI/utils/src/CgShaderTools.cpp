#ifdef USE_CG

#include "GUI/utils/CgShaderTools.h"

void
AShaderConfig::Initialize(
	CGcontext   				&_cgContext,
	const boost::filesystem::path 		&fragmentProgramFile, 
	const std::string 			&fragmentProgramName 
	)
{
	cgContext = _cgContext;

	cgGLSetDebugMode(CG_FALSE);
	cgSetParameterSettingMode( cgContext, CG_DEFERRED_PARAMETER_SETTING);

	cgFragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
	CheckForCgError("getting fragment profile ", cgContext);
	
	cgGLSetOptimalOptions( cgFragmentProfile );
	CheckForCgError("selecting fragment profile ", cgContext);

	cgFragmentProgram =
		cgCreateProgramFromFile(
		cgContext,                /* Cg runtime context */
		CG_SOURCE,                  /* Program in human-readable form */
		fragmentProgramFile.string().data(),  /* Name of file containing program */
		cgFragmentProfile,        /* Profile: OpenGL ARB vertex program */
		fragmentProgramName.data(),      /* Entry function name */
		NULL);                      /* No extra compiler options */
	CheckForCgError("creating fragment program from file ", cgContext );
	cgGLLoadProgram( cgFragmentProgram );
	CheckForCgError("loading fragment program ", cgContext );
}


void
AShaderConfig::Finalize()
{
	cgDestroyProgram(cgFragmentProgram);
}

void
AShaderConfig::Enable()
{
	cgGLBindProgram( cgFragmentProgram );
	cgGLEnableProfile( cgFragmentProfile );
}

void
AShaderConfig::Disable()
{
	cgGLDisableProfile( cgFragmentProfile );
}

void
CgBrightnessContrastShaderConfig::Initialize(
	CGcontext   				&_cgContext,
	const boost::filesystem::path 		&fragmentProgramFile, 
	const std::string 			&fragmentProgramName 
	)
{
	AShaderConfig::Initialize( _cgContext, fragmentProgramFile, fragmentProgramName );

	cgFragmentParam_Texture = cgGetNamedParameter( cgFragmentProgram, "texture" );
	CheckForCgError("getting 'texture' parameter ", cgContext );

	cgFragmentParam_BrightnessContrast = cgGetNamedParameter( cgFragmentProgram, "brightnessContrast");
	CheckForCgError("getting 'contrastBrightness' parameter ", cgContext );
}

void
CgBrightnessContrastShaderConfig::Enable()
{
	AShaderConfig::Enable();

	cgGLEnableTextureParameter( cgFragmentParam_Texture );
	cgGLSetTextureParameter( cgFragmentParam_Texture, textureName );

	cgGLSetParameter2f( cgFragmentParam_BrightnessContrast, brightnessContrast[0], brightnessContrast[1] );
}

void
CgBrightnessContrastShaderConfig::Disable()
{
	cgGLDisableTextureParameter(cgFragmentParam_Texture);
	AShaderConfig::Disable();
}

void
CgMaskBlendShaderConfig::Initialize(
	CGcontext   				&_cgContext,
	const boost::filesystem::path 		&fragmentProgramFile, 
	const std::string 			&fragmentProgramName 
	)
{

	AShaderConfig::Initialize( _cgContext, fragmentProgramFile, fragmentProgramName );

	cgFragmentParam_Texture = cgGetNamedParameter( cgFragmentProgram, "texture" );
	CheckForCgError("getting 'texture' parameter ", cgContext );

	cgFragmentParam_BlendColor = cgGetNamedParameter( cgFragmentProgram, "blendColor" );
	CheckForCgError("getting 'blendColor' parameter ", cgContext );
}

void
CgMaskBlendShaderConfig::Enable()
{
	AShaderConfig::Enable();

	cgGLEnableTextureParameter( cgFragmentParam_Texture );
	cgGLSetTextureParameter( cgFragmentParam_Texture, textureName );

	cgGLSetParameter4fv( cgFragmentParam_BlendColor, blendColor.GetData() );
}

void
CgMaskBlendShaderConfig::Disable()
{
	cgGLDisableTextureParameter(cgFragmentParam_Texture);
	AShaderConfig::Disable();
}

void
CgSimpleTransferFunctionShaderConfig::Initialize(
	CGcontext   				&_cgContext,
	const boost::filesystem::path 		&fragmentProgramFile, 
	const std::string 			&fragmentProgramName 
	)
{
	AShaderConfig::Initialize( _cgContext, fragmentProgramFile, fragmentProgramName );

	cgFragmentParam_Data = cgGetNamedParameter( cgFragmentProgram, "dataTexture" );
	CheckForCgError("getting 'dataTexture' parameter ", cgContext );

	cgFragmentParam_TransferFunction = cgGetNamedParameter( cgFragmentProgram, "transferFunctionTexture" );
	CheckForCgError("getting 'transferFunctionTexture' parameter ", cgContext );
}

void
CgSimpleTransferFunctionShaderConfig::Enable()
{
	AShaderConfig::Enable();

	cgGLEnableTextureParameter( cgFragmentParam_Data );
	cgGLSetTextureParameter( cgFragmentParam_Data, dataTexture );

	cgGLEnableTextureParameter( cgFragmentParam_TransferFunction );
	cgGLSetTextureParameter( cgFragmentParam_TransferFunction, transferFunctionTexture );
}

void
CgSimpleTransferFunctionShaderConfig::Disable()
{
	cgGLDisableTextureParameter(cgFragmentParam_Data);
	cgGLDisableTextureParameter(cgFragmentParam_TransferFunction);
	AShaderConfig::Disable();
}


void 
CheckForCgError( const std::string &situation, CGcontext &context  )
{
	CGerror error;
	const char *string = cgGetLastErrorString(&error);

	if (error != CG_NO_ERROR) {
		std::string message( TO_STRING( situation << string ) );
		if (error == CG_COMPILER_ERROR) {
			message = TO_STRING( message << "\n" << cgGetLastListing(context) );
		}
		_THROW_ CgException( message );
	}
}

#endif /*USE_CG*/

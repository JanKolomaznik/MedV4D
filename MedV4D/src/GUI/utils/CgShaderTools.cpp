#include "MedV4D/GUI/utils/CgShaderTools.h"

namespace M4D {
namespace GUI {

bool gIsCgInitialized = false;
CGcontext gCgContext;

void
InitializeCg()
{//TODO - check if design with global variables is proper one
	if ( gIsCgInitialized ) {
		LOG( "Cg already initialized" );
		return;
	}
	gCgContext = cgCreateContext();
	CheckForCgError("creating context");
	cgGLSetDebugMode( CG_FALSE );
	cgSetParameterSettingMode(gCgContext, CG_IMMEDIATE_PARAMETER_SETTING/*CG_DEFERRED_PARAMETER_SETTING*/);
	cgGLRegisterStates(gCgContext);
	CheckForCgError("registering standard CgFX states");
	cgGLSetManageTextureParameters(gCgContext, CG_TRUE);
	CheckForCgError("manage texture parameters");

	gIsCgInitialized = true;
	LOG( "Cg initialized" );
}

void
FinalizeCg()
{
	cgDestroyContext( gCgContext );
}


void
CgEffect::Initialize(/*CGcontext   				&cgContext,*/
			const boost::filesystem::path 		&effectFile
			)
{
	mCgEffect = cgCreateEffectFromFile( gCgContext, effectFile.string().data(), NULL );
	CheckForCgError( TO_STRING("creating cg effect from file \"" << effectFile << "\"." ) );

	LOG( "Cg effect \"" << effectFile.filename() << "\" loaded" );

	CGtechnique cgTechnique = cgGetFirstTechnique(mCgEffect);
	while (cgTechnique) {
		if ( cgValidateTechnique(cgTechnique) == CG_FALSE ) {
			LOG( "\tTechnique " << cgGetTechniqueName(cgTechnique) << " did not validate. Skipping." );
		} else {

			LOG( "\tTechnique " << cgGetTechniqueName(cgTechnique) << " validated. Enabling." );
			mCgTechniques[ cgGetTechniqueName(cgTechnique) ] = cgTechnique;
		}
		cgTechnique = cgGetNextTechnique( cgTechnique );
	}
	if ( mCgTechniques.size() == 0 ) {
		throw CgException( "No technique validated!" );
	}
}


void
CgEffect::Finalize()
{
	cgDestroyEffect(mCgEffect);
}

void
CgEffect::SetParameter( std::string aName, const GLTextureImage &aTexture )
{
	SetTextureParameter( aName, aTexture.GetTextureGLID() );
}

void
CgEffect::SetParameter( std::string aName, const GLTextureImage3D &aImage )
{
	SetTextureParameter( TO_STRING( aName << ".data" ), aImage.GetTextureGLID() );

	SetParameter( TO_STRING( aName << ".size" ), aImage.getExtents().maximum - aImage.getExtents().minimum );

	SetParameter( TO_STRING( aName << ".realSize" ), aImage.getExtents().realMaximum - aImage.getExtents().realMinimum );

	SetParameter( TO_STRING( aName << ".realMinimum" ), aImage.getExtents().realMinimum );

	SetParameter( TO_STRING( aName << ".realMaximum" ), aImage.getExtents().realMaximum );
}

void
CgEffect::SetParameter( std::string aName, const GLTransferFunctionBuffer1D &aTransferFunction )
{
	SetTextureParameter( TO_STRING( aName << ".data" ), aTransferFunction.GetTextureID() );

	SetParameter( TO_STRING( aName << ".interval" ), aTransferFunction.GetMappedInterval() );

	SetParameter( TO_STRING( aName << ".sampleCount" ), aTransferFunction.GetSampleCount() );
}


void
CgEffect::SetTextureParameter( std::string aName, GLuint aTexture )
{
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect, aName.data() );
//	ASSERT( )	TODO check type;

	cgGLSetupSampler( cgParameter, aTexture );
	//cgSetSamplerState( cgParameter );
}

void
CgEffect::SetParameter( std::string aName, float aValue )
{
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect, aName.data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValuefr( cgParameter, 1, &aValue );
}

void
CgEffect::SetParameter( std::string aName, double aValue )
{
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect, aName.data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValuedr( cgParameter, 1, &aValue );
}

void
CgEffect::SetParameter( std::string aName, int aValue )
{
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect, aName.data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValueir( cgParameter, 1, &aValue );
}

void
CgEffect::SetParameter( std::string aName, const glm::fmat4x4 &aMatrix )
{
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect, aName.data() );
	/*LOG( "baseType = " << cgGetTypeString( cgGetParameterBaseType( cgParameter ) ) );
	LOG( "namedType = " << cgGetTypeString( cgGetParameterNamedType( cgParameter ) ) );*/

	cgSetParameterValuefr( cgParameter, 16, glm::value_ptr( aMatrix ) );
	CheckForCgError("set matrix parameter");
	
	/*glm::fmat4x4 tmp;
	cgGetParameterValuefr( cgParameter, 16, glm::value_ptr( tmp ) );
	LOG( aName << ":\n" << aMatrix << "tmp:\n" << tmp << "\n" );*/
}

void
CgEffect::SetParameter( std::string aName, const glm::dmat4x4 &aMatrix )
{
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect, aName.data() );
	/*LOG( "baseType = " << cgGetTypeString( cgGetParameterBaseType( cgParameter ) ) );
	LOG( "namedType = " << cgGetTypeString( cgGetParameterNamedType( cgParameter ) ) );*/

	cgSetParameterValuedr( cgParameter, 16, glm::value_ptr( aMatrix ) );
	CheckForCgError("set matrix parameter");
	
	/*glm::dmat4x4 tmp;
	cgGetParameterValuedr( cgParameter, 16, glm::value_ptr( tmp ) );
	LOG( aName << ":\n" << aMatrix << "tmp:\n" << tmp << "\n" );*/
}

void
CgEffect::SetParameter( std::string aName, const BoundingBox3D &aValue )
{
	CGparameter cgParameter = cgGetNamedEffectParameter( mCgEffect, TO_STRING( aName << ".vertices" ).data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValuefr( cgParameter, 3*8, aValue.vertices[0].GetData() );
}

void
CgEffect::SetParameter( std::string aName, const float *aValue, size_t aCount )
{
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect, aName.data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValuefr( cgParameter, static_cast<int>(aCount), aValue );
}

void
CgEffect::SetParameter( std::string aName, const double *aValue, size_t aCount )
{
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect, aName.data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValuedr( cgParameter, static_cast<int>(aCount), aValue );
}

void
CgEffect::SetParameter( std::string aName, const int *aValue, size_t aCount )
{
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect, aName.data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValueir( cgParameter, static_cast<int>(aCount), aValue );
}

void
CgEffect::SetParameter( std::string aName, const M4D::Planef &aPlane )
{
	SetParameter( TO_STRING( aName << ".point" ), static_cast< const Vector3f &>( aPlane.point() ) );

	SetParameter( TO_STRING( aName << ".normal" ), aPlane.normal() );
}

void
CgEffect::SetParameter( std::string aName, const M4D::GLViewSetup &aViewSetup )
{
	SetParameter( TO_STRING( aName << ".modelViewProj" ), glm::fmat4x4(aViewSetup.modelViewProj) );
	SetParameter( TO_STRING( aName << ".modelMatrix" ), glm::fmat4x4(aViewSetup.model) );
	SetParameter( TO_STRING( aName << ".projMatrix" ), glm::fmat4x4(aViewSetup.projection) );
	SetParameter( TO_STRING( aName << ".viewMatrix" ), glm::fmat4x4(aViewSetup.view) );
}


void
CgEffect::SetGLStateMatrixParameter( std::string aName, CGGLenum matrix, CGGLenum transform )
{
	CGparameter cgParameter = cgGetNamedEffectParameter( mCgEffect, aName.data() );

	cgGLSetStateMatrixParameter(cgParameter, matrix, transform);
}




void
CgEffect::prepareState()
{

}

//******************************************************************

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
	CheckForCgError(TO_STRING("creating fragment program from file \"" << fragmentProgramFile.string() << "\"." ), cgContext );
	cgGLLoadProgram( cgFragmentProgram );
	CheckForCgError(TO_STRING("loading fragment program \"" << fragmentProgramName << "\"." ), cgContext );
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

	cgFragmentParam_Texture = cgGetNamedParameter( cgFragmentProgram, "dataTexture" );
	CheckForCgError("getting 'dataTexture' parameter ", cgContext );

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
//*********************************************************************************
//*********************************************************************************
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

//*********************************************************************************
//*********************************************************************************
void
CgTransferFunctionShadingShaderConfig::Initialize(
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

	cgFragmentParam_LightPosition = cgGetNamedParameter( cgFragmentProgram, "lightPosition" );
	CheckForCgError("getting 'lightPosition' parameter ", cgContext );

	cgFragmentParam_EyePosition = cgGetNamedParameter( cgFragmentProgram, "eyePosition" );
	CheckForCgError("getting 'eyePosition' parameter ", cgContext );

	cgFragmentParam_SliceSpacing = cgGetNamedParameter( cgFragmentProgram, "sliceSpacing" );
	CheckForCgError("getting 'sliceSpacing' parameter ", cgContext );

	cgFragmentParam_SliceNormal = cgGetNamedParameter( cgFragmentProgram, "sliceNormal" );
	CheckForCgError("getting 'sliceNormal' parameter ", cgContext );
}

void
CgTransferFunctionShadingShaderConfig::Enable()
{
	AShaderConfig::Enable();

	cgGLEnableTextureParameter( cgFragmentParam_Data );
	cgGLSetTextureParameter( cgFragmentParam_Data, dataTexture );

	cgGLEnableTextureParameter( cgFragmentParam_TransferFunction );
	cgGLSetTextureParameter( cgFragmentParam_TransferFunction, transferFunctionTexture );

	cgGLSetParameter3f( cgFragmentParam_EyePosition, eyePosition[0], eyePosition[1], eyePosition[2] );

	cgGLSetParameter3f( cgFragmentParam_LightPosition, lightPosition[0], lightPosition[1], lightPosition[2] );

	cgGLSetParameter3f( cgFragmentParam_SliceNormal, sliceNormal[0], sliceNormal[1], sliceNormal[2] );

	cgGLSetParameter1f( cgFragmentParam_SliceSpacing, sliceSpacing );
}

void
CgTransferFunctionShadingShaderConfig::Disable()
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
/*
void
GLDrawVolumeSlicesForVertexShader(
		M4D::BoundingBox3D	bbox,
		Camera			camera,
		unsigned 		numberOfSteps,
		float			cutPlane
		)
{
	Vector< float, 3> vertices[6];

	float 				min = 0; 
	float 				max = 0;
	unsigned			minId = 0;	
	unsigned			maxId = 0;	
	GetBBoxMinMaxDistance( 
		bbox, 
		camera.GetEyePosition(), 
		camera.GetCenterDirection(), 
		min, 
		max,
		minId,	
		maxId	
		);
	
	float stepSize = cutPlane * (max - min) / numberOfSteps;
	Vector< float, 3> planePoint = camera.GetEyePosition() + camera.GetCenterDirection() * max;
	for( unsigned i = 0; i < numberOfSteps; ++i ) {
		//Obtain intersection of the optical axis and the currently rendered plane
		planePoint -= stepSize * camera.GetCenterDirection();
		//Get n-gon as intersection of the current plane and bounding box
		unsigned count = M4D::GetPlaneVerticesInBoundingBox( 
				bbox, planePoint, camera.GetCenterDirection(), minId, vertices
				);

		//Render n-gon
		glBegin( GL_POLYGON );
			for( unsigned j = 0; j < 6; ++j ) {
				glVertex2i( j, i );
			}
		glEnd();
	}

}*/

} //namespace M4D
} //namespace GUI

#include "MedV4D/GUI/utils/CgShaderTools.h"
#include <boost/bind.hpp>

namespace M4D {
namespace GUI {

bool gIsCgInitialized = false;
CGcontext gCgContext;

void
initializeCg()
{//TODO - check if design with global variables is proper one
	if ( gIsCgInitialized ) {
		LOG( "Cg already initialized" );
		return;
	}
	gCgContext = cgCreateContext();
	checkForCgError("creating context");
	cgGLSetDebugMode( CG_FALSE );
	cgSetParameterSettingMode(gCgContext, /*CG_IMMEDIATE_PARAMETER_SETTING*/CG_DEFERRED_PARAMETER_SETTING);
	cgGLRegisterStates(gCgContext);
	checkForCgError("registering standard CgFX states");
	cgGLSetManageTextureParameters(gCgContext, CG_TRUE);
	checkForCgError("manage texture parameters");

	gIsCgInitialized = true;
	LOG( "Cg initialized" );
}

void
finalizeCg()
{
	cgDestroyContext( gCgContext );
}

void
CgFXShader::initialize(//CGcontext   				&cgContext,
			const boost::filesystem::path 		&effectFile
			)
{
	mEffectName = effectFile.filename().string();

	if ( !boost::filesystem::is_regular_file( effectFile ) ) {
		_THROW_ CgException( mEffectName, boost::str( boost::format( "Effect could not be loaded! `%1%` is not regular file." ) %effectFile ) );
	}
	mCgEffect = makeResourceGuardPtr< CGeffect >( boost::bind<CGeffect>( &cgCreateEffectFromFile, gCgContext, effectFile.string().data(), static_cast<const char **>(0) ), boost::bind<void>( &cgDestroyEffect, _1 ) );

	checkForCgError( TO_STRING("creating cg effect from file \"" << effectFile << "\"." ) );

	LOG( "Cg effect \"" << effectFile.filename() << "\" loaded" );

	CGtechnique cgTechnique = cgGetFirstTechnique(mCgEffect->get());
	if ( !cgTechnique ) {
		_THROW_ CgException( mEffectName, "No technique found!" );
	}
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
		_THROW_ CgException( mEffectName, "No technique validated!" );
	}
	mEffectInitialized = true;
}


void
CgFXShader::finalize()
{
	if (mEffectInitialized) {
		mEffectInitialized = false;
		mCgEffect.reset();
	}
}
/*
void
CgEffect::Initialize(//CGcontext   				&cgContext,
			const boost::filesystem::path 		&effectFile
			)
{
	mEffectName = effectFile.filename().string();

	if ( !boost::filesystem::is_regular_file( effectFile ) ) {
		_THROW_ CgException( mEffectName, boost::str( boost::format( "Effect could not be loaded! `%1%` is not regular file." ) %effectFile ) );
	}
	mCgEffect = makeResourceGuardPtr< CGeffect >( boost::bind<CGeffect>( &cgCreateEffectFromFile, gCgContext, effectFile.string().data(), static_cast<const char **>(0) ), boost::bind<void>( &cgDestroyEffect, _1 ) );

	CheckForCgError( TO_STRING("creating cg effect from file \"" << effectFile << "\"." ) );

	LOG( "Cg effect \"" << effectFile.filename() << "\" loaded" );

	CGtechnique cgTechnique = cgGetFirstTechnique(mCgEffect->get());
	if ( !cgTechnique ) {
		_THROW_ CgException( mEffectName, "No technique found!" );
	}
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
		_THROW_ CgException( mEffectName, "No technique validated!" );
	}
	mEffectInitialized = true;
}


void
CgEffect::Finalize()
{
	if (mEffectInitialized) {
		mEffectInitialized = false;
		mCgEffect.reset();
	}
}

void
CgEffect::SetParameter( std::string aName, const GLTextureImage &aTexture )
{
	ASSERT(isInitialized());
	SetTextureParameter( aName, aTexture.GetTextureGLID() );
}

void
CgEffect::SetParameter( std::string aName, const GLTextureImage3D &aImage )
{
	ASSERT(isInitialized());
	SetTextureParameter( TO_STRING( aName << ".data" ), aImage.GetTextureGLID() );

	SetParameter( TO_STRING( aName << ".size" ), aImage.getExtents().maximum - aImage.getExtents().minimum );

	SetParameter( TO_STRING( aName << ".realSize" ), aImage.getExtents().realMaximum - aImage.getExtents().realMinimum );

	SetParameter( TO_STRING( aName << ".realMinimum" ), aImage.getExtents().realMinimum );

	SetParameter( TO_STRING( aName << ".realMaximum" ), aImage.getExtents().realMaximum );
}

void
CgEffect::SetParameter( std::string aName, const GLTransferFunctionBuffer1D &aTransferFunction )
{
	ASSERT(isInitialized());
	SetTextureParameter( TO_STRING( aName << ".data" ), aTransferFunction.getTextureID() );

	SetParameter( TO_STRING( aName << ".interval" ), aTransferFunction.getMappedInterval() );

	SetParameter( TO_STRING( aName << ".sampleCount" ), aTransferFunction.getSampleCount() );
}


void
CgEffect::SetTextureParameter( std::string aName, GLuint aTexture )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect->get(), aName.data() );
//	ASSERT( )	TODO check type;

	cgGLSetupSampler( cgParameter, aTexture );
	//cgSetSamplerState( cgParameter );
}

void
CgEffect::SetParameter( std::string aName, float aValue )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect->get(), aName.data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValuefr( cgParameter, 1, &aValue );
}

void
CgEffect::SetParameter( std::string aName, double aValue )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect->get(), aName.data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValuedr( cgParameter, 1, &aValue );
}

void
CgEffect::SetParameter( std::string aName, int aValue )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect->get(), aName.data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValueir( cgParameter, 1, &aValue );
}

void
CgEffect::SetParameter( std::string aName, const glm::fmat4x4 &aMatrix )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect->get(), aName.data() );
	//LOG( "baseType = " << cgGetTypeString( cgGetParameterBaseType( cgParameter ) ) );
	//LOG( "namedType = " << cgGetTypeString( cgGetParameterNamedType( cgParameter ) ) );

	cgSetParameterValuefr( cgParameter, 16, glm::value_ptr( aMatrix ) );
	CheckForCgError("set matrix parameter");
	
	//glm::fmat4x4 tmp;
	//cgGetParameterValuefr( cgParameter, 16, glm::value_ptr( tmp ) );
	//LOG( aName << ":\n" << aMatrix << "tmp:\n" << tmp << "\n" );
}

void
CgEffect::SetParameter( std::string aName, const glm::dmat4x4 &aMatrix )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect->get(), aName.data() );
	//LOG( "baseType = " << cgGetTypeString( cgGetParameterBaseType( cgParameter ) ) );
	//LOG( "namedType = " << cgGetTypeString( cgGetParameterNamedType( cgParameter ) ) );

	cgSetParameterValuedr( cgParameter, 16, glm::value_ptr( aMatrix ) );
	CheckForCgError("set matrix parameter");
	
	//glm::dmat4x4 tmp;
	//cgGetParameterValuedr( cgParameter, 16, glm::value_ptr( tmp ) );
	//LOG( aName << ":\n" << aMatrix << "tmp:\n" << tmp << "\n" );
}

void
CgEffect::SetParameter( std::string aName, const BoundingBox3D &aValue )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter( mCgEffect->get(), TO_STRING( aName << ".vertices" ).data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValuefr( cgParameter, 3*8, aValue.vertices[0].GetData() );
}

void
CgEffect::SetParameter( std::string aName, const float *aValue, size_t aCount )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect->get(), aName.data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValuefr( cgParameter, static_cast<int>(aCount), aValue );
}

void
CgEffect::SetParameter( std::string aName, const double *aValue, size_t aCount )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect->get(), aName.data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValuedr( cgParameter, static_cast<int>(aCount), aValue );
}

void
CgEffect::SetParameter( std::string aName, const int *aValue, size_t aCount )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect->get(), aName.data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValueir( cgParameter, static_cast<int>(aCount), aValue );
}

void
CgEffect::SetParameter( std::string aName, const M4D::Planef &aPlane )
{
	ASSERT(isInitialized());
	SetParameter( TO_STRING( aName << ".point" ), static_cast< const Vector3f &>( aPlane.point() ) );

	SetParameter( TO_STRING( aName << ".normal" ), aPlane.normal() );
}

void
CgEffect::SetParameter( std::string aName, const M4D::GLViewSetup &aViewSetup )
{
	ASSERT(isInitialized());
	SetParameter( TO_STRING( aName << ".modelViewProj" ), glm::fmat4x4(aViewSetup.modelViewProj) );
	SetParameter( TO_STRING( aName << ".modelMatrix" ), glm::fmat4x4(aViewSetup.model) );
	SetParameter( TO_STRING( aName << ".projMatrix" ), glm::fmat4x4(aViewSetup.projection) );
	SetParameter( TO_STRING( aName << ".viewMatrix" ), glm::fmat4x4(aViewSetup.view) );
}


void
CgEffect::SetGLStateMatrixParameter( std::string aName, CGGLenum matrix, CGGLenum transform )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter( mCgEffect->get(), aName.data() );

	cgGLSetStateMatrixParameter(cgParameter, matrix, transform);
}




void
CgEffect::prepareState()
{

}*/



void 
checkForCgError( const std::string &situation, CGcontext &context  )
{
	CGerror error;
	const char *string = cgGetLastErrorString(&error);

	if (error != CG_NO_ERROR) {
		std::string message( TO_STRING( situation << string ) );
		const char * listing = cgGetLastListing(context);
		if( listing ) {
			message = TO_STRING( message << "\nLast listing:" << listing );
		}
		_THROW_ CgException( message );
	}
}


} //namespace M4D
} //namespace GUI

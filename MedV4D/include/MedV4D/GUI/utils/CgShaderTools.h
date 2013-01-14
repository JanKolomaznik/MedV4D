#ifndef CG_SHADER_TOOLS_H
#define CG_SHADER_TOOLS_H

#include "MedV4D/GUI/utils/OGLDrawing.h"
#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgGL.h>
#include <string>
#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/GeometricPrimitives.h"
#include "MedV4D/GUI/utils/DrawingTools.h"
#include "MedV4D/GUI/utils/Camera.h"
#include "MedV4D/GUI/utils/GLTextureImage.h"
#include "MedV4D/GUI/utils/TransferFunctionBuffer.h"
#include <map>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "MedV4D/Common/RAII.h"

#include <boost/shared_ptr.hpp>

namespace M4D {
namespace GUI {

extern bool gIsCgInitialized;
extern CGcontext gCgContext;

void
InitializeCg();

void
FinalizeCg();

class CgException: public M4D::ErrorHandling::ExceptionBase
{
public:
	CgException( std::string aEffectName, std::string aReport ) throw() : ExceptionBase( TO_STRING( aEffectName << ": " << aReport ) ) {}
	CgException( std::string aReport ) throw() : ExceptionBase( aReport ) {}
	~CgException() throw(){}
};

void 
CheckForCgError( const std::string &situation, CGcontext &context = gCgContext );


class CgEffect
{
public:
	CgEffect(): mEffectInitialized(false)
	{}

	void
	Initialize(	/*CGcontext   				&cgContext,*/
			const boost::filesystem::path 		&effectFile
			);

	void
	Finalize();

	template< typename TGeometryRenderFunctor >
	void
	ExecuteTechniquePass( std::string aTechniqueName, TGeometryRenderFunctor aDrawGeometry );

	template< size_t Dim >
	void
	SetParameter( std::string aName, const Vector<float, Dim> &value );

	template< size_t Dim >
	void
	SetParameter( std::string aName, const Vector<double, Dim> &value );

	template< size_t Dim >
	void
	SetParameter( std::string aName, const Vector<unsigned int, Dim> &value );

	template< size_t Dim >
	void
	SetParameter( std::string aName, const Vector<int, Dim> &value );

	void
	SetParameter( std::string aName, const GLTextureImage &aTexture );

	void
	SetParameter( std::string aName, const GLTextureImage3D &aImage );

	void
	SetParameter( std::string aName, const GLTransferFunctionBuffer1D &aTransferFunction );

	void
	SetParameter( std::string aName, float aValue );

	void
	SetParameter( std::string aName, double aValue );

	void
	SetParameter( std::string aName, int aValue );

	void
	SetParameter( std::string aName, const float *aValue, size_t aCount );

	void
	SetParameter( std::string aName, const double *aValue, size_t aCount );

	void
	SetParameter( std::string aName, const int *aValue, size_t aCount );
	
	void
	SetParameter( std::string aName, const glm::fmat4x4 &aMatrix );
	
	void
	SetParameter( std::string aName, const glm::dmat4x4 &aMatrix );

	void
	SetParameter( std::string aName, const BoundingBox3D &aValue );
	
	void
	SetTextureParameter( std::string aName, GLuint aTexture );

	void
	SetParameter( std::string aName, const M4D::Planef &aPlane );
	
	void
	SetParameter( std::string aName, const M4D::GLViewSetup &aViewSetup );

	void
	SetGLStateMatrixParameter( std::string aName, CGGLenum matrix, CGGLenum transform );

	bool
	isInitialized() const 
	{ return mEffectInitialized; }
protected:
	virtual void
	prepareState();

	boost::shared_ptr< ResourceGuard< CGeffect > >	mCgEffect;
	std::map< std::string, CGtechnique >	mCgTechniques;
	std::string	mEffectName;

	bool mEffectInitialized;
};

template< typename TGeometryRenderFunctor >
void
CgEffect::ExecuteTechniquePass( std::string aTechniqueName, TGeometryRenderFunctor aDrawGeometry )
{
	if (!isInitialized()) {
		_THROW_ EObjectNotInitialized();
	}
	M4D::GLPushAtribs pushAttribs; // GL_CHECKED_CALL( glPushAttrib( GL_ALL_ATTRIB_BITS ) );

	prepareState();
	
	std::map< std::string, CGtechnique >::iterator it = mCgTechniques.find( aTechniqueName );
	if ( it == mCgTechniques.end() ) {
		_THROW_ CgException( mEffectName, TO_STRING( "Unavailable technique : " << aTechniqueName ) );
	}
	CGtechnique cgTechnique = it->second;

	CGpass pass = cgGetFirstPass( cgTechnique );
	CheckForCgError( TO_STRING( "getting first pass for technique " << cgGetTechniqueName( cgTechnique ) ) );
	while ( pass ) {
		cgSetPassState( pass );

		aDrawGeometry();

		cgResetPassState( pass );
		pass = cgGetNextPass( pass );
	}
	CheckForGLError( "After effect application :" );

	//GL_CHECKED_CALL( glPopAttrib() );
}

template< size_t Dim >
void
CgEffect::SetParameter( std::string aName, const Vector<float, Dim> &value )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter( mCgEffect->get(), aName.data() );

//	ASSERT( )	TODO check type;
	cgSetParameterValuefr( cgParameter, Dim, value.GetData() );	
}

template< size_t Dim >
void
CgEffect::SetParameter( std::string aName, const Vector<double, Dim> &value )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter( mCgEffect->get(), aName.data() );

//	ASSERT( )	TODO check type;
	cgSetParameterValuedr( cgParameter, Dim, value.GetData() );	
}

template< size_t Dim >
void
CgEffect::SetParameter( std::string aName, const Vector<unsigned int, Dim> &value )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter( mCgEffect->get(), aName.data() );

//	ASSERT( )	TODO check type;
	cgSetParameterValueir( cgParameter, Dim, reinterpret_cast< const int* >( value.GetData() ) );	
}

template< size_t Dim >
void
CgEffect::SetParameter( std::string aName, const Vector<int, Dim> &value )
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter( mCgEffect->get(), aName.data() );

//	ASSERT( )	TODO check type;
	cgSetParameterValueir( cgParameter, Dim, value.GetData() );	
}

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
GLDrawVolumeSlicesForVertexShader(
		M4D::BoundingBox3D	bbox,
		Camera		camera,
		unsigned 	numberOfSteps,
		float		cutPlane = 1.0f
		);


} //namespace M4D
} //namespace GUI

#endif /*CG_SHADER_TOOLS_H*/

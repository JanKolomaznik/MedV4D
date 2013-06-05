#ifndef CG_SHADER_TOOLS_H
#define CG_SHADER_TOOLS_H

#include <soglu/CgFXShader.hpp>

#ifdef DISABLE_0

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
initializeCg();

void
finalizeCg();

class CgException: public M4D::ErrorHandling::ExceptionBase
{
public:
	CgException( std::string aEffectName, std::string aReport ) throw() : ExceptionBase( TO_STRING( aEffectName << ": " << aReport ) ) {}
	CgException( std::string aReport ) throw() : ExceptionBase( aReport ) {}
	~CgException() throw(){}
};

void 
checkForCgError( const std::string &situation, CGcontext &context = gCgContext );


class CgFXShader 
{
public:

	void
	initialize(const boost::filesystem::path &effectFile);

	void
	finalize();

	template<typename TParameterType>
	void
	setParameter(std::string aName, const TParameterType &aValue);

	template<typename TParameterType>
	void
	setParameter(std::string aName, const TParameterType *aValue, size_t aCount);
	
	//TODO - modify
	void
	setParameter(std::string aName, const M4D::GLViewSetup &aViewSetup);
	
	void
	setParameter(std::string aName, const BoundingBox3D &aValue);
	
	void
	setTextureParameter(std::string aName, GLuint aTexture);

	void
	setParameter(std::string aName, const M4D::Planef &aPlane);
	
	void
	setParameter(std::string aName, const GLTextureImage &aTexture);

	void
	setParameter(std::string aName, const GLTextureImage3D &aImage);

	void
	setParameter(std::string aName, const GLTransferFunctionBuffer1D &aTransferFunction);
	
	void
	setParameter(std::string aName, const Camera &aCamera);
	
	template< typename TGeometryRenderFunctor >
	void
	executeTechniquePass( std::string aTechniqueName, TGeometryRenderFunctor aDrawGeometry );
	
	bool
	isInitialized() const 
	{ return mEffectInitialized; }
protected:

	boost::shared_ptr< ResourceGuard< CGeffect > >	mCgEffect;
	std::map< std::string, CGtechnique >	mCgTechniques;
	std::string	mEffectName;

	bool mEffectInitialized;
};

namespace detail {


	inline void
	parameterSetter(CGparameter aParameter, float aValue)
	{
		cgSetParameterValuefr(aParameter, 1, &aValue);
	}

	inline void
	parameterSetter(CGparameter aParameter, double aValue)
	{
		cgSetParameterValuedr(aParameter, 1, &aValue);
	}

	inline void
	parameterSetter(CGparameter aParameter, int aValue)
	{
		cgSetParameterValueir(aParameter, 1, &aValue);
	}


	inline void
	parameterSetter(CGparameter aParameter, const glm::fmat4x4 &aMatrix)
	{
		cgSetParameterValuefr(aParameter, 16, glm::value_ptr( aMatrix ));
	}

	inline void
	parameterSetter(CGparameter aParameter, const glm::dmat4x4 &aMatrix)
	{
		cgSetParameterValuedr(aParameter, 16, glm::value_ptr( aMatrix ));
	}

	//-------------------------------------------------------------------------

	inline void
	parameterSetter(CGparameter aParameter, const float *aValue, size_t aSize)
	{
		cgSetParameterValuefr(aParameter, aSize, aValue);
	}

	inline void
	parameterSetter(CGparameter aParameter, const double *aValue, size_t aSize)
	{
		cgSetParameterValuedr(aParameter, aSize, aValue);
	}

	inline void
	parameterSetter(CGparameter aParameter, const int *aValue, size_t aSize)
	{
		cgSetParameterValueir(aParameter, aSize, aValue);
	}
	
	//-------------------------------------------------------------------------

	inline void
	parameterSetter(CGparameter aParameter, const glm::fvec3 &aVec3)
	{
		cgSetParameterValuefr(aParameter, 3, glm::value_ptr( aVec3 ));
	}

	inline void
	parameterSetter(CGparameter aParameter, const glm::dvec3 &aVec3)
	{
		cgSetParameterValuedr(aParameter, 3, glm::value_ptr( aVec3 ));
	}

	//-------------------------------------------------------------------------
	template<size_t tDim>
	inline void
	parameterSetter(CGparameter aParameter, const Vector<float, tDim> &aValue)
	{
		cgSetParameterValuefr(aParameter, tDim, aValue.GetData());
	}
	
	template<size_t tDim>
	inline void
	parameterSetter(CGparameter aParameter, const Vector<int, tDim> &aValue)
	{
		cgSetParameterValueir(aParameter, tDim, aValue.GetData());
	}


} //namespace detail


template<typename TParameterType>
void
CgFXShader::setParameter(std::string aName, const TParameterType &aValue)
{
	//SOGLU_ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter( mCgEffect->get(), aName.data() );

	detail::parameterSetter(cgParameter, aValue);
}

template<typename TParameterType>
void
CgFXShader::setParameter(std::string aName, const TParameterType *aValue, size_t aCount)
{
	//SOGLU_ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter( mCgEffect->get(), aName.data() );

	detail::parameterSetter(cgParameter, aValue, aCount);
}

inline void
CgFXShader::setParameter( std::string aName, const M4D::GLViewSetup &aViewSetup )
{
	ASSERT(isInitialized());
	setParameter(aName + ".modelViewProj", glm::fmat4x4(aViewSetup.modelViewProj) );
	setParameter(aName + ".modelMatrix", glm::fmat4x4(aViewSetup.model) );
	setParameter(aName + ".projMatrix", glm::fmat4x4(aViewSetup.projection) );
	setParameter(aName + ".viewMatrix", glm::fmat4x4(aViewSetup.view) );
}

inline void
CgFXShader::setParameter( std::string aName, const M4D::Planef &aPlane )
{
	ASSERT(isInitialized());
	setParameter(aName + ".point", static_cast< const Vector3f &>( aPlane.point() ) );

	setParameter(aName + ".normal", aPlane.normal() );
}

inline void
CgFXShader::setParameter( std::string aName, const GLTextureImage &aTexture )
{
	ASSERT(isInitialized());
	setTextureParameter( aName, aTexture.GetTextureGLID() );
}

inline void
CgFXShader::setParameter( std::string aName, const GLTextureImage3D &aImage )
{
	ASSERT(isInitialized());
	setTextureParameter(aName + ".data", aImage.GetTextureGLID() );

	setParameter(aName + ".size", aImage.getExtents().maximum - aImage.getExtents().minimum );

	setParameter(aName + ".realSize", aImage.getExtents().realMaximum - aImage.getExtents().realMinimum );

	setParameter(aName + ".realMinimum", aImage.getExtents().realMinimum );

	setParameter(aName + ".realMaximum", aImage.getExtents().realMaximum );
}

inline void
CgFXShader::setParameter(std::string aName, const GLTransferFunctionBuffer1D &aTransferFunction )
{
	ASSERT(isInitialized());
	setTextureParameter(aName + ".data", aTransferFunction.getTextureID() );

	setParameter(aName + ".interval", aTransferFunction.getMappedInterval() );

	setParameter(aName + ".sampleCount", aTransferFunction.getSampleCount() );
}

inline void
CgFXShader::setParameter(std::string aName, const Camera &aCamera)
{
	setParameter(aName + ".eyePosition", aCamera.eyePosition());
	
	setParameter(aName + ".viewDirection", aCamera.targetDirection());
	
	setParameter(aName + ".upDirection", aCamera.upDirection());
}

inline void
CgFXShader::setParameter(std::string aName, const BoundingBox3D &aValue)
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter( mCgEffect->get(), TO_STRING( aName << ".vertices" ).data() );
//	ASSERT( )	TODO check type;

	cgSetParameterValuefr( cgParameter, 3*8, &(aValue.vertices[0].x) );
}

inline void
CgFXShader::setTextureParameter(std::string aName, GLuint aTexture)
{
	ASSERT(isInitialized());
	CGparameter cgParameter = cgGetNamedEffectParameter(mCgEffect->get(), aName.data());
//	ASSERT( )	TODO check type;

	cgGLSetupSampler( cgParameter, aTexture );
	//cgSetSamplerState( cgParameter );
}

template< typename TGeometryRenderFunctor >
void
CgFXShader::executeTechniquePass( std::string aTechniqueName, TGeometryRenderFunctor aDrawGeometry )
{
	if (!isInitialized()) {
		_THROW_ EObjectNotInitialized();
	}
	M4D::GLPushAtribs pushAttribs; // GL_CHECKED_CALL( glPushAttrib( GL_ALL_ATTRIB_BITS ) );

	
	std::map< std::string, CGtechnique >::iterator it = mCgTechniques.find( aTechniqueName );
	if ( it == mCgTechniques.end() ) {
		_THROW_ CgException( mEffectName, TO_STRING( "Unavailable technique : " << aTechniqueName ) );
	}
	CGtechnique cgTechnique = it->second;

	CGpass pass = cgGetFirstPass( cgTechnique );
	checkForCgError( TO_STRING( "getting first pass for technique " << cgGetTechniqueName( cgTechnique ) ) );
	while ( pass ) {
		cgSetPassState( pass );

		aDrawGeometry();

		cgResetPassState( pass );
		pass = cgGetNextPass( pass );
	}
	CheckForGLError( "After effect application :" );

	//GL_CHECKED_CALL( glPopAttrib() );
}


typedef CgFXShader CgEffect;


} //namespace M4D
} //namespace GUI

#endif /*CG_SHADER_TOOLS_H*/
#endif //DISABLE_0
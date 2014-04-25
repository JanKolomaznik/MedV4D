#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/GeometricPrimitives.h"
//#include "MedV4D/GUI/utils/CgShaderTools.h"
#include <soglu/CgFXShader.hpp>

//#include "MedV4D/GUI/utils/GLTextureImage.h"
#include <soglu/GLTextureImage.hpp>
#include <boost/bind.hpp>
//#include "MedV4D/GUI/utils/TransferFunctionBuffer.h"
#include <vorgl/TransferFunctionBuffer.hpp>
#include <vorgl/VolumeRenderer.hpp>

#include "MedV4D/GUI/renderers/RendererTools.h"

namespace M4D
{
namespace GUI
{
namespace Renderer
{

enum RenderingFlags
{
	rf_NONE        = 0,

	rf_SHADING     = 1,
	rf_JITTERING     = 1 << 1,

	rf_PREINTEGRATED     = 1 << 2,
	rf_INTERPOLATION     = 1 << 3
};


extern boost::filesystem::path gVolumeRendererShaderPath;

struct VolumeRestrictions
{
	VolumeRestrictions(): resX( 0.0f, 1.0f ), resY( 0.0f, 1.0f ), resZ( 0.0f, 1.0f )
	{}
	VolumeRestrictions( const Vector2f &aX, const Vector2f &aY, const Vector2f &aZ ): resX( aX ), resY( aY ), resZ( aZ )
	{}

	void
	get( Vector2f &aX, Vector2f &aY, Vector2f &aZ )const
	{
		aX = resX;
		aY = resY;
		aZ = resZ;
	}

	void
	get3D( Vector3f &i1, Vector3f &i2 )const
	{
		i1[0] = resX[0];
		i1[1] = resY[0];
		i1[2] = resZ[0];

		i2[0] = resX[1];
		i2[1] = resY[1];
		i2[2] = resZ[1];
	}

	Vector2f resX, resY, resZ;
};


class VolumeRenderer: public vorgl::VolumeRenderer
{
public:
	//typedef std::map< std::wstring, int > ColorTransformNameToIDMap;
	//typedef std::map< int, std::wstring > ColorTransformIDToNameMap;

	struct RenderingConfiguration;

	void
	initialize()
	{
		vorgl::VolumeRenderer::initialize(gVolumeRendererShaderPath);

		mAvailableColorTransforms.clear();
		//mAvailableColorTransforms.push_back( WideNameIdPair( L"Transfer function", ctTransferFunction1D ) );
		//mAvailableColorTransforms.push_back( WideNameIdPair( L"MIP", ctMaxIntensityProjection ) );
		mAvailableColorTransforms.push_back( ColorTransformNameIDList::value_type( "Transfer function", ctTransferFunction1D ) );
		mAvailableColorTransforms.push_back( ColorTransformNameIDList::value_type( "MIP", ctMaxIntensityProjection ) );
		mAvailableColorTransforms.push_back( ColorTransformNameIDList::value_type( "Basic", ctBasic ) );
		mAvailableColorTransforms.push_back( ColorTransformNameIDList::value_type( "Test", ctTestColorTransform ) );
	}

	void
	reloadShaders()
	{
		vorgl::VolumeRenderer::loadShaders(gVolumeRendererShaderPath);
	}

	/*void
	Finalize();*/

	virtual void
	Render( RenderingConfiguration & aConfig, const soglu::GLViewSetup &aViewSetup );

	const ColorTransformNameIDList&
	GetAvailableColorTransforms()const
	{
		return mAvailableColorTransforms;
	}
protected:


	ColorTransformNameIDList		mAvailableColorTransforms;
};

struct VolumeRenderer::RenderingConfiguration
{
	RenderingConfiguration()
		: //primaryImageData( NULL ),
		//secondaryImageData( NULL ),
		colorTransform( ctMaxIntensityProjection ),
		jitterEnabled( true ),
		jitterStrength( 1.0f ),
		shadingEnabled( true ),
		integralTFEnabled( false ),
		sampleCount( 150 ),
		enableInterpolation(true ),
		enableVolumeRestrictions( false ),
		enableCutPlane( false ),
		cutPlaneCameraTargetOffset( 0.0f ),
		multiDatasetRenderingStyle( mdrsOnlyPrimary )
	{ }
	soglu::GLTextureImage3D::WPtr		primaryImageData;
	soglu::GLTextureImage3D::WPtr		secondaryImageData;

	int					colorTransform;
	vorgl::GLTransferFunctionBuffer1D::ConstWPtr	transferFunction;
	vorgl::GLTransferFunctionBuffer1D::ConstWPtr	integralTransferFunction;
	glm::fvec2				lutWindow;
	soglu::Camera				camera;
	bool					jitterEnabled;
	float					jitterStrength;
	bool					shadingEnabled;
	bool					integralTFEnabled;
	size_t					sampleCount;
	bool					enableInterpolation;

	glm::fvec3				lightPosition;

	bool					enableVolumeRestrictions;
	VolumeRestrictions			volumeRestrictions;

	bool					enableCutPlane;
	soglu::Planef				cutPlane;
	float					cutPlaneCameraTargetOffset;

	MultiDatasetRenderingStyle		multiDatasetRenderingStyle;

	soglu::TextureId depthBuffer;
	glm::ivec2 windowSize;
};


}//Renderer
}//GUI
}//M4D

#endif /*VOLUME_RENDERER_H*/

#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/GeometricPrimitives.h"
#include "MedV4D/GUI/utils/CgShaderTools.h"
#include "MedV4D/GUI/utils/GLTextureImage.h"
#include <boost/bind.hpp>
#include "MedV4D/GUI/utils/TransferFunctionBuffer.h"
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


class VolumeRenderer
{
public:
	//typedef std::map< std::wstring, int > ColorTransformNameToIDMap;
	//typedef std::map< int, std::wstring > ColorTransformIDToNameMap;

	struct RenderingConfiguration;

	void
	Initialize();

	void
	Finalize();
	
	virtual void
	Render( RenderingConfiguration & aConfig, const GLViewSetup &aViewSetup );

	const ColorTransformNameIDList&
	GetAvailableColorTransforms()const
	{
		return mAvailableColorTransforms;
	}
protected:
	
	void
	basicRendering( 
		const Camera &aCamera, 
		const GLTextureImageTyped<3> &aImage, 
		const M4D::BoundingBox3D &aBoundingBox, 
		size_t aSliceCount,
		bool aJitterEnabled,
		float aJitterStrength, 
		bool aEnableCutPlane,
		Planef aCutPlane,
		bool aEnableInterpolation,
		Vector2f aLutWindow,
		const GLViewSetup &aViewSetup,
		bool aMIP,
		uint64 aFlags
     	);
	
	void
	transferFunctionRendering( 
		const Camera &aCamera, 
		const GLTextureImageTyped<3> &aImage, 
		const M4D::BoundingBox3D &aBoundingBox, 
		size_t aSliceCount, 
		bool aJitterEnabled,
		float aJitterStrength, 
		bool aEnableCutPlane,
		Planef aCutPlane,
		bool aEnableInterpolation,
		const GLViewSetup &aViewSetup,
		const GLTransferFunctionBuffer1D &aTransferFunction,
		Vector3f aLightPosition,
		uint64 aFlags
	);
	
	void
	setupJittering(float aJitterStrength);
	
	void
	setupView(const Camera &aCamera, const GLViewSetup &aViewSetup);
	
	void
	setupSamplingProcess(const M4D::BoundingBox3D &aBoundingBox, const Camera &aCamera, size_t aSliceCount);
	
	void
	setupLights(const Vector3f &aLightPosition);
	
	void
	initJitteringTexture();

	void
	reallocateArrays( size_t aNewMaxSampleCount )
	{
		if( mVertices ) {
			delete [] mVertices;
		}
		if( mIndices ) {
			delete [] mIndices;
		}

		mVertices = new glm::fvec3[ (aNewMaxSampleCount+1) * 6 ];
		mIndices = new unsigned[ (aNewMaxSampleCount+1) * 7 ];
		mMaxSampleCount = aNewMaxSampleCount;
	}

	CGcontext   				mCgContext;
	CgEffect				mCgEffect;
	GLuint					mNoiseMap;

	ColorTransformNameIDList		mAvailableColorTransforms;

	glm::fvec3 *mVertices;
	unsigned *mIndices;
	size_t		mMaxSampleCount;
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
	GLTextureImage3D::WPtr			primaryImageData;
	GLTextureImage3D::WPtr			secondaryImageData;
	
	int					colorTransform;
	GLTransferFunctionBuffer1D::ConstWPtr	transferFunction;
	GLTransferFunctionBuffer1D::ConstWPtr	integralTransferFunction;
	Vector2f				lutWindow;
	Camera					camera;
	bool					jitterEnabled;
	float					jitterStrength;
	bool					shadingEnabled;
	bool					integralTFEnabled;
	size_t					sampleCount;				
	bool					enableInterpolation;

	Vector3f				lightPosition;

	bool					enableVolumeRestrictions;
	VolumeRestrictions			volumeRestrictions;

	bool					enableCutPlane;
	Planef					cutPlane;
	float					cutPlaneCameraTargetOffset;
	
	MultiDatasetRenderingStyle		multiDatasetRenderingStyle;
};


}//Renderer
}//GUI
}//M4D

#endif /*VOLUME_RENDERER_H*/

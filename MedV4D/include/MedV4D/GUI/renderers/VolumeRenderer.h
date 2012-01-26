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

		mVertices = new Vector3f[ (aNewMaxSampleCount+1) * 6 ];
		mIndices = new unsigned[ (aNewMaxSampleCount+1) * 7 ];
		mMaxSampleCount = aNewMaxSampleCount;
	}

	CGcontext   				mCgContext;
	CgEffect				mCgEffect;
	GLuint					mNoiseMap;

	ColorTransformNameIDList		mAvailableColorTransforms;

	Vector3f	*mVertices;
	unsigned	*mIndices;
	size_t		mMaxSampleCount;
};

struct VolumeRenderer::RenderingConfiguration
{
	RenderingConfiguration()
		: imageData( NULL ), colorTransform( ctMaxIntensityProjection ), 
		transferFunction( NULL ), integralTransferFunction( NULL ), jitterEnabled( true ), jitterStrength( 1.0f ), shadingEnabled( true ), 
		integralTFEnabled( false ), sampleCount( 150 ), enableInterpolation(true ), enableVolumeRestrictions( false ), enableCutPlane( false ), cutPlaneCameraTargetOffset( 0.0f )
	{ }
	const GLTextureImage3D			*imageData;
	
	int					colorTransform;
	const GLTransferFunctionBuffer1D	*transferFunction;
	const GLTransferFunctionBuffer1D	*integralTransferFunction;
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
};


}//Renderer
}//GUI
}//M4D

#endif /*VOLUME_RENDERER_H*/

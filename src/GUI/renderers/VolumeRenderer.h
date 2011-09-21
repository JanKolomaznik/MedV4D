#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

#include "common/Common.h"
#include "common/GeometricPrimitives.h"
#include "GUI/utils/CgShaderTools.h"
#include "GUI/utils/GLTextureImage.h"
#include <boost/bind.hpp>
#include "GUI/utils/TransferFunctionBuffer.h"
#include "GUI/renderers/RendererTools.h"

namespace M4D
{
namespace GUI
{
namespace Renderer
{

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
	Render( RenderingConfiguration & aConfig, bool aSetupView = true );

	const ColorTransformNameIDList&
	GetAvailableColorTransforms()const
	{
		return mAvailableColorTransforms;
	}
protected:
	void
	initJitteringTexture();

	void
	reallocateArrays( unsigned aNewMaxSampleCount )
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
	unsigned	mMaxSampleCount;
};

struct VolumeRenderer::RenderingConfiguration
{
	RenderingConfiguration()
		: imageData( NULL ), colorTransform( ctMaxIntensityProjection ), 
		transferFunction( NULL ), jitterEnabled( true ), shadingEnabled( true ), 
		sampleCount( 150 ), enableVolumeRestrictions( false ), enableCutPlane( false ), cutPlaneCameraTargetOffset( 0.0f )
	{ }
	const GLTextureImage3D			*imageData;
	
	int					colorTransform;
	const GLTransferFunctionBuffer1D	*transferFunction;
	Vector2f				lutWindow;
	Camera					camera;
	bool					jitterEnabled;
	bool					shadingEnabled;
	size_t					sampleCount;				

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

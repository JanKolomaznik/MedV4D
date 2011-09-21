#include "GUI/renderers/VolumeRenderer.h"

namespace M4D
{
namespace GUI
{
namespace Renderer
{

void
applyVolumeRestrictionsOnBoundingBox( M4D::BoundingBox3D &aBBox, const VolumeRestrictions &aVolumeRestrictions )
{
	Vector3f corner1 = aBBox.getMin();
	Vector3f corner2 = aBBox.getMax();
	Vector3f size = corner2 - corner1;
	
	Vector3f i1, i2;
	aVolumeRestrictions.get3D( i1, i2 );

	//LOG( "Restrictions : " << i1 << " - " << i2 );

	corner2 = corner1 + VectorMemberProduct( i2, size );
	corner1 += VectorMemberProduct( i1, size );
	aBBox = M4D::BoundingBox3D( corner1, corner2 );
}

void
VolumeRenderer::Initialize()
{
	InitializeCg();
	mCgEffect.Initialize( "ImageRender.cgfx" );

	initJitteringTexture();

	mMaxSampleCount = 0;
	mVertices = NULL;
	mIndices = NULL;

	mAvailableColorTransforms.clear();
	//mAvailableColorTransforms.push_back( WideNameIdPair( L"Transfer function", ctTransferFunction1D ) );
	//mAvailableColorTransforms.push_back( WideNameIdPair( L"MIP", ctMaxIntensityProjection ) );
	mAvailableColorTransforms.push_back( ColorTransformNameIDList::value_type( "Transfer function", ctTransferFunction1D ) );
	mAvailableColorTransforms.push_back( ColorTransformNameIDList::value_type( "MIP", ctMaxIntensityProjection ) );
	mAvailableColorTransforms.push_back( ColorTransformNameIDList::value_type( "Basic", ctBasic ) );
}

void
VolumeRenderer::initJitteringTexture()
{
	//TODO make better - destroy
	int size = 32;
	uint8 * buf = new uint8[size*size];
	srand( (unsigned)time(NULL) );
	for( int i = 0; i < size*size; ++i ) {
		buf[i] = static_cast<uint8>( 255.0f * rand()/(float)RAND_MAX );
	}
	glGenTextures(1, &mNoiseMap );
	//glActiveTextureARB(GL_TEXTURE3_ARB);
	glBindTexture( GL_TEXTURE_2D, mNoiseMap );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_LUMINANCE8,
			size,
			size,
			0, 
			GL_LUMINANCE, 
			GL_UNSIGNED_BYTE,
			buf
		    );
	glBindTexture( GL_TEXTURE_2D, 0 );

	delete buf;
}

void
VolumeRenderer::Finalize()
{
	//TODO
}
//#define HHHHAAAA
#ifdef HHHHAAAA
void
VolumeRenderer::Render( VolumeRenderer::RenderingConfiguration & aConfig, bool aSetupView )
{
	ASSERT( aConfig.imageData != NULL );

	static int edgeOrder[8*12] = {
		 10, 11,  9,  4,  8,  5,  1,  0,  6,  2,  3,  7,
		 11,  8, 10,  5,  9,  6,  2,  1,  7,  3,  0,  4,
		  8,  9, 11,  6, 10,  7,  3,  2,  4,  0,  1,  5,
		  9, 10,  8,  7, 11,  4,  0,  3,  5,  1,  2,  6,
		  1,  0,  2,  4,  3,  7, 10, 11,  6,  9,  8,  5,
		  2,  1,  3,  5,  0,  4, 11,  8,  7, 10,  9,  6,
		  3,  2,  0,  6,  1,  5,  8,  9,  4, 11, 10,  7,
		  0,  3,  1,  7,  2,  6,  9, 10,  5,  8, 11,  4
	};


	if( aSetupView ) {
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		//Set viewing parameters
		M4D::SetViewAccordingToCamera( aConfig.camera );
	}
	glMatrixMode(GL_MODELVIEW);
	
	unsigned sliceCount = aConfig.sampleCount;
	if( sliceCount > mMaxSampleCount ) {
		reallocateArrays( sliceCount );
	}

	float renderingSliceThickness = 1.0f;

	M4D::BoundingBox3D bbox( aConfig.imageData->GetMinimum(), aConfig.imageData->GetMaximum() );
	if ( aConfig.enableVolumeRestrictions ) {
		applyVolumeRestrictionsOnBoundingBox( bbox, aConfig.volumeRestrictions );
	}
	float 				min = 0; 
	float 				max = 0;
	unsigned			minId = 0;	
	unsigned			maxId = 0;
	GetBBoxMinMaxDistance( 
			bbox, 
			aConfig.camera.GetEyePosition(), 
			aConfig.camera.GetTargetDirection(), 
			min, 
			max, 
		       	minId,	
		       	maxId	
			);

	mCgEffect.SetParameter( "gImageData3D", *aConfig.imageData );
	mCgEffect.SetParameter( "gMappedIntervalBands", aConfig.imageData->GetMappedInterval() );
	mCgEffect.SetParameter( "gLightPosition", aConfig.lightPosition );
	mCgEffect.SetParameter( "gLightColor", Vector3f( 1.0f, 1.0f, 1.0f ) );
	mCgEffect.SetParameter( "gEyePosition", aConfig.camera.GetEyePosition() );
	mCgEffect.SetParameter( "gRenderingSliceThickness", renderingSliceThickness );

	mCgEffect.SetParameter( "gViewDirection", aConfig.camera.GetTargetDirection() );
	mCgEffect.SetParameter( "edgeOrder", edgeOrder, 8*12 );
	mCgEffect.SetParameter( "gMinID", (int)minId );
	mCgEffect.SetParameter( "gBBox", bbox );

	Vector3f tmp = VectorMemberDivision( aConfig.camera.GetTargetDirection(), aConfig.imageData->GetSize() );
	mCgEffect.SetParameter( "gSliceNormalTexCoords", tmp );
	mCgEffect.SetTextureParameter( "gNoiseMap", mNoiseMap );
	mCgEffect.SetParameter( "gNoiseMapSize", Vector2f( 32.0f, 32.0f ) );

	mCgEffect.SetParameter( "gEnableCutPlane", aConfig.enableCutPlane );
	mCgEffect.SetParameter( "gCutPlane", aConfig.cutPlane );

	mCgEffect.SetGLStateMatrixParameter( "gModelViewProj", CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY );
	
	std::string techniqueName;
	switch ( aConfig.colorTransform ) {
	case ctTransferFunction1D:
		{
			mCgEffect.SetTextureParameter( "gTransferFunction1D", aConfig.transferFunction->GetTextureID() );
			mCgEffect.SetParameter( "gTransferFunction1DInterval", aConfig.transferFunction->GetMappedInterval() );

			if ( aConfig.jitterEnabled ) {
				if ( aConfig.shadingEnabled ) {
					techniqueName = "TransferFunction1DShadingJitter_3D";
				} else {
					techniqueName = "TransferFunction1DJitter_3D";
				}
			} else {
				if ( aConfig.shadingEnabled ) {
					techniqueName = "TransferFunction1DShading_3D";
				} else {
					techniqueName = "TransferFunction1D_3D";
				}
			}
		}
		break;
	case ctMaxIntensityProjection:
		{
			mCgEffect.SetParameter( "gWLWindow", aConfig.lutWindow );
			techniqueName = "WLWindowMIP_3D";
		}
		break;
	default:
		ASSERT( false );
	}
	//D_PRINT(  aConfig.imageData->GetMinimum() << " ----- " << aConfig.imageData->GetMaximum() << "++++" << sliceCount );
	mCgEffect.ExecuteTechniquePass(
			techniqueName, 
			boost::bind( &M4D::GLDrawVolumeSliceCenterSamples, 
				bbox,
				aConfig.camera,
				sliceCount,
				1.0f
				) 
			); 

	M4D::CheckForGLError( "OGL error : " );
}

#else

void
VolumeRenderer::Render( VolumeRenderer::RenderingConfiguration & aConfig, bool aSetupView )
{
	ASSERT( aConfig.imageData != NULL );


	if( aSetupView ) {
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		//Set viewing parameters
		M4D::SetViewAccordingToCamera( aConfig.camera );
	}
	glMatrixMode(GL_MODELVIEW);
	
	unsigned sliceCount = aConfig.sampleCount;
	if( sliceCount > mMaxSampleCount ) {
		reallocateArrays( sliceCount );
	}
	float renderingSliceThickness = 1.0f;
	M4D::BoundingBox3D bbox( aConfig.imageData->GetMinimum(), aConfig.imageData->GetMaximum() );
	if ( aConfig.enableVolumeRestrictions ) {
		applyVolumeRestrictionsOnBoundingBox( bbox, aConfig.volumeRestrictions );
	}

	//aConfig.enableCutPlane = true;
	//aConfig.cutPlane = Planef( bbox.getCenter(), Vector3f( 0.3f, 0.45f, 0.1f ) );

	mCgEffect.SetParameter( "gImageData3D", *aConfig.imageData );
	mCgEffect.SetParameter( "gMappedIntervalBands", aConfig.imageData->GetMappedInterval() );
	mCgEffect.SetParameter( "gLightPosition", Vector3f( 3000.0f, 3000.0f, -3000.0f ) );
	mCgEffect.SetParameter( "gLightColor", Vector3f( 1.0f, 1.0f, 1.0f ) );
	mCgEffect.SetParameter( "gEyePosition", aConfig.camera.GetEyePosition() );
	mCgEffect.SetParameter( "gRenderingSliceThickness", renderingSliceThickness );

	Vector3f tmp = VectorMemberDivision( aConfig.camera.GetTargetDirection(), aConfig.imageData->GetSize() );
	mCgEffect.SetParameter( "gSliceNormalTexCoords", tmp );
	mCgEffect.SetTextureParameter( "gNoiseMap", mNoiseMap );
	mCgEffect.SetParameter( "gNoiseMapSize", Vector2f( 32.0f, 32.0f ) );

	mCgEffect.SetParameter( "gEnableCutPlane", aConfig.enableCutPlane );
	mCgEffect.SetParameter( "gCutPlane", aConfig.cutPlane );

	mCgEffect.SetGLStateMatrixParameter( "gModelViewProj", CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY );

	std::string techniqueName;
	switch ( aConfig.colorTransform ) {
	case ctTransferFunction1D:
		{
			mCgEffect.SetTextureParameter( "gTransferFunction1D", aConfig.transferFunction->GetTextureID() );
			mCgEffect.SetParameter( "gTransferFunction1DInterval", aConfig.transferFunction->GetMappedInterval() );

			if ( aConfig.jitterEnabled ) {
				if ( aConfig.shadingEnabled ) {
					techniqueName = "TransferFunction1DShadingJitter_3D";
				} else {
					techniqueName = "TransferFunction1DJitter_3D";
				}
			} else {
				if ( aConfig.shadingEnabled ) {
					techniqueName = "TransferFunction1DShading_3D";
				} else {
					techniqueName = "TransferFunction1D_3D";
				}
			}
		}
		break;
	case ctMaxIntensityProjection:
		{
			mCgEffect.SetParameter( "gWLWindow", aConfig.lutWindow );
			techniqueName = "WLWindowMIP_3D";
		}
		break;
	case ctBasic:
		{
			mCgEffect.SetParameter( "gWLWindow", aConfig.lutWindow );
			techniqueName = "WLWindowBasic_3D";
		}
		break;
	default:
		ASSERT( false );
	}
	//D_PRINT(  aConfig.imageData->GetMinimum() << " ----- " << aConfig.imageData->GetMaximum() << "++++" << sliceCount );
	mCgEffect.ExecuteTechniquePass(
			techniqueName, 
			boost::bind( &M4D::GLDrawVolumeSlices_Buffered, 
				bbox,
				aConfig.camera,
				sliceCount,
				mVertices,
				mIndices,
				1.0f
				) 
			); 

	/*mCgEffect.ExecuteTechniquePass(
			techniqueName, 
			boost::bind( &M4D::GLDrawVolumeSlices, 
				bbox,
				aConfig.camera,
				sliceCount,
				1.0f
				) 
			); */

	M4D::CheckForGLError( "OGL error : " );
}
#endif

}//Renderer
}//GUI
}//M4D


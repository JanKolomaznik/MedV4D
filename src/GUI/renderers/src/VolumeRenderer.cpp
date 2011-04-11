#include "GUI/renderers/VolumeRenderer.h"

namespace M4D
{
namespace GUI
{
namespace Renderer
{

void
VolumeRenderer::Initialize()
{
	InitializeCg();
	mCgEffect.Initialize( "ImageRender.cgfx" );

	int size = 32;
	uint8 * buf = new uint8[size*size];
	srand( (unsigned)time(NULL) );
	for( int i = 0; i < size*size; ++i ) {
		buf[i] = 255.0f * rand()/(float)RAND_MAX;
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

/*void
VolumeRenderer::Render( VolumeRenderer::RenderingConfiguration & aConfig, bool aSetupView )
{
	ASSERT( aConfig.imageData != NULL );

	if( aSetupView ) {
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		//Set viewing parameters
		M4D::SetViewAccordingToCamera( aConfig.camera );
		glMatrixMode(GL_MODELVIEW);
	}
	
	unsigned sliceCount = aConfig.sampleCount;
	float renderingSliceThickness = 1.0f;

	glColor3f( 1.0f, 0.0f, 0.0f );
	M4D::GLDrawBoundingBox( aConfig.imageData->GetMinimum(), aConfig.imageData->GetMaximum() );

	mCgEffect.SetParameter( "gImageData3D", *aConfig.imageData );
	mCgEffect.SetParameter( "gMappedIntervalBands", aConfig.imageData->GetMappedInterval() );
	mCgEffect.SetParameter( "gLightPosition", Vector3f( 3000.0f, 3000.0f, -3000.0f ) );
	mCgEffect.SetParameter( "gLightColor", Vector3f( 1.0f, 1.0f, 1.0f ) );
	mCgEffect.SetParameter( "gEyePosition", aConfig.camera.GetEyePosition() );
	mCgEffect.SetParameter( "gRenderingSliceThickness", renderingSliceThickness );

	mCgEffect.SetParameter( "gViewDirection", aConfig.camera.GetTargetDirection() );

	Vector3f tmp = VectorMemberDivision( aConfig.camera.GetTargetDirection(), aConfig.imageData->GetSize() );
	mCgEffect.SetParameter( "gSliceNormalTexCoords", tmp );
	mCgEffect.SetTextureParameter( "gNoiseMap", mNoiseMap );
	mCgEffect.SetParameter( "gNoiseMapSize", Vector2f( 32.0f, 32.0f ) );


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
	M4D::SetVolumeTextureCoordinateGeneration( aConfig.imageData->GetMinimum(), aConfig.imageData->GetRealSize() );
	mCgEffect.ExecuteTechniquePass(
			techniqueName, 
			boost::bind( &M4D::GLDrawVolumeSliceCenterSamples, 
				M4D::BoundingBox3D( aConfig.imageData->GetMinimum(), aConfig.imageData->GetMaximum() ),
				aConfig.camera,
				sliceCount,
				1.0f
				) 
			); 

	M4D::DisableVolumeTextureCoordinateGeneration();
	M4D::CheckForGLError( "OGL error : " );
}*/

void
VolumeRenderer::Render( VolumeRenderer::RenderingConfiguration & aConfig, bool aSetupView )
{
	ASSERT( aConfig.imageData != NULL );


	if( aSetupView ) {
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		//Set viewing parameters
		M4D::SetViewAccordingToCamera( aConfig.camera );
		glMatrixMode(GL_MODELVIEW);
	}
	
	unsigned sliceCount = aConfig.sampleCount;
	float renderingSliceThickness = 1.0f;

	glColor3f( 1.0f, 0.0f, 0.0f );
	M4D::GLDrawBoundingBox( aConfig.imageData->GetMinimum(), aConfig.imageData->GetMaximum() );

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
	M4D::SetVolumeTextureCoordinateGeneration( aConfig.imageData->GetMinimum(), aConfig.imageData->GetRealSize() );
	mCgEffect.ExecuteTechniquePass(
			techniqueName, 
			boost::bind( &M4D::GLDrawVolumeSlices, 
				M4D::BoundingBox3D( aConfig.imageData->GetMinimum(), aConfig.imageData->GetMaximum() ),
				aConfig.camera,
				sliceCount,
				1.0f
				) 
			); 

	M4D::DisableVolumeTextureCoordinateGeneration();
	M4D::CheckForGLError( "OGL error : " );
}


}//Renderer
}//GUI
}//M4D


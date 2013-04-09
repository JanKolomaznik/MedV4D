#include "MedV4D/GUI/renderers/VolumeRenderer.h"

#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/verbose_operator.hpp>

namespace M4D
{
namespace GUI
{
namespace Renderer
{

boost::filesystem::path gVolumeRendererShaderPath;
  
void
applyVolumeRestrictionsOnBoundingBox( M4D::BoundingBox3D &aBBox, const VolumeRestrictions &aVolumeRestrictions )
{
	glm::fvec3 corner1 = aBBox.getMin();
	glm::fvec3 corner2 = aBBox.getMax();
	glm::fvec3 size = corner2 - corner1;
	
	Vector3f i1Tmp, i2Tmp;
	aVolumeRestrictions.get3D(i1Tmp, i2Tmp);

	glm::fvec3 i1(i1Tmp[0], i1Tmp[1], i1Tmp[2]);
	glm::fvec3 i2(i2Tmp[0], i2Tmp[1], i2Tmp[2]);
	//LOG( "Restrictions : " << i1 << " - " << i2 );

	corner2 = corner1 + (i2 * size);
	corner1 += (i1 * size);
	aBBox = M4D::BoundingBox3D( corner1, corner2 );
}

void
VolumeRenderer::Initialize()
{
	initializeCg();
	mCgEffect.initialize( gVolumeRendererShaderPath/*"ImageRender.cgfx"*/ );

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

enum TFConfigurationFlags{
	tfShading	= 1,
	tfJittering	= 1 << 1,
	tfIntegral	= 1 << 2
};

void
VolumeRenderer::Render( VolumeRenderer::RenderingConfiguration & aConfig, const GLViewSetup &aViewSetup )
{
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

	GLTextureImageTyped<3>::Ptr primaryData = aConfig.primaryImageData.lock();
	if ( !primaryData ) {
		_THROW_ ErrorHandling::EObjectUnavailable( "Primary texture not available" );
	}
	
	mCgEffect.setParameter( "gPrimaryImageData3D", *primaryData );
	mCgEffect.setParameter( "gMappedIntervalBands", primaryData->GetMappedInterval() );
	
	GLTextureImageTyped<3>::Ptr secondaryData = aConfig.secondaryImageData.lock();
	if( secondaryData ) {
		mCgEffect.setParameter( "gSecondaryImageData3D", *secondaryData );
	}
	
	size_t sliceCount = aConfig.sampleCount;
	if( sliceCount > mMaxSampleCount ) {
		reallocateArrays( sliceCount );
	}
	M4D::BoundingBox3D bbox( toGLM(primaryData->getExtents().realMinimum), toGLM(primaryData->getExtents().realMaximum) );
	if ( aConfig.enableVolumeRestrictions ) {
		applyVolumeRestrictionsOnBoundingBox( bbox, aConfig.volumeRestrictions );
	}
	//LOG( bbox );
	float 				min = 0; 
	float 				max = 0;
	unsigned			minId = 0;	
	unsigned			maxId = 0;
	getBBoxMinMaxDistance( 
			bbox, 
			aConfig.camera.GetEyePosition(), 
			aConfig.camera.GetTargetDirection(), 
			min, 
			max, 
		       	minId,	
		       	maxId	
			);
	float renderingSliceThickness = (max-min)/static_cast< float >( sliceCount );

	mCgEffect.setParameter( "gLight.position", aConfig.lightPosition );
	mCgEffect.setParameter( "gLight.color", Vector3f( 1.0f, 1.0f, 1.0f ) );
	mCgEffect.setParameter( "gLight.ambient", Vector3f( 0.3f, 0.3f, 0.3f ) );
	mCgEffect.setParameter( "gEyePosition", aConfig.camera.GetEyePosition() );
	mCgEffect.setParameter( "gRenderingSliceThickness", renderingSliceThickness );

	mCgEffect.setParameter( "gViewDirection", aConfig.camera.GetTargetDirection() );
	mCgEffect.setParameter( "edgeOrder", edgeOrder, 8*12 );
	mCgEffect.setParameter( "gMinID", (int)minId );
	mCgEffect.setParameter( "gBBox", bbox );

	Vector3f tmp = VectorMemberDivision( fromGLM(aConfig.camera.GetTargetDirection()), primaryData->getExtents().realMaximum-primaryData->getExtents().realMinimum);
	mCgEffect.setParameter( "gSliceNormalTexCoords", tmp );
	mCgEffect.setTextureParameter( "gNoiseMap", mNoiseMap );
	mCgEffect.setParameter( "gNoiseMapSize", Vector2f( 32.0f, 32.0f ) );
	mCgEffect.setParameter( "gJitterStrength", aConfig.jitterStrength  );

	mCgEffect.setParameter( "gEnableCutPlane", aConfig.enableCutPlane );
	mCgEffect.setParameter( "gCutPlane", aConfig.cutPlane );
	mCgEffect.setParameter( "gEnableInterpolation", aConfig.enableInterpolation );

	//mCgEffect.SetGLStateMatrixParameter( "gModelViewProj", CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY );
	mCgEffect.setParameter( "gViewSetup", aViewSetup );
	
	std::string techniqueName;
	GLTransferFunctionBuffer1D::ConstPtr transferFunction;
	GLTransferFunctionBuffer1D::ConstPtr integralTransferFunction;
	switch ( aConfig.colorTransform ) {
	case ctTransferFunction1D:
		{
			transferFunction = aConfig.transferFunction.lock();
			if ( !transferFunction ) {
				_THROW_ M4D::ErrorHandling::EObjectUnavailable( "Transfer function no available" );
			}

			mCgEffect.setParameter( "gTransferFunction1D", *transferFunction );

			integralTransferFunction = aConfig.integralTransferFunction.lock();
			if( integralTransferFunction ) {
				mCgEffect.setParameter( "gIntegralTransferFunction1D", *integralTransferFunction );
			}
			unsigned configurationMask = 0;

			if ( aConfig.jitterEnabled ) configurationMask |= tfJittering;
			if ( aConfig.shadingEnabled ) configurationMask |= tfShading;
			if ( aConfig.integralTFEnabled ) configurationMask |= tfIntegral;

			switch ( configurationMask ) {
			case 0:
			case tfIntegral:
				techniqueName = "TransferFunction1D_3D";
				break;
			case tfJittering:
			case ( tfJittering | tfIntegral ):
				techniqueName = "TransferFunction1DJitter_3D";
				break;
			case ( tfShading | tfIntegral ):
				techniqueName = "IntegralTransferFunction1DShading_3D";
				break;
			case ( tfJittering | tfShading | tfIntegral ):
				techniqueName = "IntegralTransferFunction1DShadingJitter_3D";
				break;
			case ( tfShading ):
				techniqueName = "TransferFunction1DShading_3D";
				break;
			case ( tfJittering | tfShading ):
				techniqueName = "TransferFunction1DShadingJitter_3D";
				break;
			default:
				ASSERT( false );
			}
			/*if ( aConfig.jitterEnabled ) {
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
			}*/
		}
		break;
	case ctMaxIntensityProjection:
		{
			mCgEffect.setParameter( "gWLWindow", aConfig.lutWindow );
			techniqueName = "WLWindowMIP_3D";
		}
		break;
	case ctBasic:
		{
			mCgEffect.setParameter( "gWLWindow", aConfig.lutWindow );
			techniqueName = "WLWindowBasic_3D";
		}
		break;
	default:
		ASSERT( false );
	}
	//D_PRINT(  aConfig.imageData->GetMinimum() << " ----- " << aConfig.imageData->GetMaximum() << "++++" << sliceCount );
	mCgEffect.executeTechniquePass(
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
			boost::bind( &M4D::GLDrawVolumeSliceCenterSamples, 
				bbox,
				aConfig.camera,
				sliceCount,
				1.0f
				) 
			);*/

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

}//Renderer
}//GUI
}//M4D


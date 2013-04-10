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


void
VolumeRenderer::setupView(const Camera &aCamera, const GLViewSetup &aViewSetup)
{
	mCgEffect.setParameter("gCamera", aCamera);
	mCgEffect.setParameter("gViewSetup", aViewSetup );
	//mCgEffect.SetGLStateMatrixParameter( "gModelViewProj", CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY );
}

void
VolumeRenderer::setupLights(const Vector3f &aLightPosition)
{
	mCgEffect.setParameter( "gLight.position", aLightPosition );
	mCgEffect.setParameter( "gLight.color", Vector3f( 1.0f, 1.0f, 1.0f ) );
	mCgEffect.setParameter( "gLight.ambient", Vector3f( 0.3f, 0.3f, 0.3f ) );
}

void
VolumeRenderer::setupJittering(float aJitterStrength)
{
	mCgEffect.setTextureParameter( "gNoiseMap", mNoiseMap );
	mCgEffect.setParameter("gNoiseMapSize", Vector2f( 32.0f, 32.0f ) );
	mCgEffect.setParameter("gJitterStrength", aJitterStrength  );
}

void
VolumeRenderer::setupSamplingProcess(const M4D::BoundingBox3D &aBoundingBox, const Camera &aCamera, size_t aSliceCount)
{
	/*static int edgeOrder[8*12] = {
		 10, 11,  9,  4,  8,  5,  1,  0,  6,  2,  3,  7,
		 11,  8, 10,  5,  9,  6,  2,  1,  7,  3,  0,  4,
		  8,  9, 11,  6, 10,  7,  3,  2,  4,  0,  1,  5,
		  9, 10,  8,  7, 11,  4,  0,  3,  5,  1,  2,  6,
		  1,  0,  2,  4,  3,  7, 10, 11,  6,  9,  8,  5,
		  2,  1,  3,  5,  0,  4, 11,  8,  7, 10,  9,  6,
		  3,  2,  0,  6,  1,  5,  8,  9,  4, 11, 10,  7,
		  0,  3,  1,  7,  2,  6,  9, 10,  5,  8, 11,  4
	};
	mCgEffect.setParameter( "edgeOrder", edgeOrder, 8*12 );
	mCgEffect.setParameter( "gMinID", (int)minId );
	mCgEffect.setParameter( "gBBox", aBoundingBox );*/
	
	float 		min = 0; 
	float 		max = 0;
	unsigned		minId = 0;	
	unsigned		maxId = 0;
	getBBoxMinMaxDistance( 
			aBoundingBox, 
			aCamera.eyePosition(), 
			aCamera.targetDirection(), 
			min, 
			max, 
		       	minId,	
		       	maxId	
			);
	float renderingSliceThickness = (max-min)/static_cast< float >( aSliceCount );

	mCgEffect.setParameter("gRenderingSliceThickness", renderingSliceThickness);
}



enum TFConfigurationFlags{
	tfShading	= 1,
	tfJittering	= 1 << 1,
	tfIntegral	= 1 << 2
};

namespace detail {
	
static const uint64 FLAGS_TO_NAME_SUFFIXES_MASK = rf_SHADING | rf_JITTERING | rf_PREINTEGRATED;
static const std::string gFlagsToNameSuffixes[] = 
	{
		std::string("Simple"),
		std::string("Shading"),
		std::string("Jittering"),
		std::string("JitteringShading"),
		std::string("PreintegratedSimple"),
		std::string("PreintegratedShading"),
		std::string("PreintegratedJittering"),
		std::string("PreintegratedJitteringShading"),
		
		std::string("UNDEFINED_COMBINATION")
	};
	
}

void
VolumeRenderer::Render( VolumeRenderer::RenderingConfiguration & aConfig, const GLViewSetup &aViewSetup )
{
	GLTextureImageTyped<3>::Ptr primaryData = aConfig.primaryImageData.lock();
	if ( !primaryData ) {
		_THROW_ ErrorHandling::EObjectUnavailable( "Primary texture not available" );
	}
	
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
	
	uint64 flags = 0;
	flags |= aConfig.jitterEnabled ? rf_JITTERING : 0;
	
	switch ( aConfig.colorTransform ) {
	case ctTransferFunction1D:
		{
			GLTransferFunctionBuffer1D::ConstPtr transferFunction;
			GLTransferFunctionBuffer1D::ConstPtr integralTransferFunction;
			transferFunction = aConfig.transferFunction.lock();
			if ( !transferFunction ) {
				_THROW_ M4D::ErrorHandling::EObjectUnavailable( "Transfer function no available" );
			}

			integralTransferFunction = aConfig.integralTransferFunction.lock();
			if( integralTransferFunction ) {
				mCgEffect.setParameter( "gIntegralTransferFunction1D", *integralTransferFunction );
			}
			
			flags |= aConfig.integralTFEnabled ? rf_PREINTEGRATED : 0;
			flags |= aConfig.shadingEnabled ? rf_SHADING : 0;
			
			transferFunctionRendering( 
				aConfig.camera, 
				*primaryData, 
				bbox, 
				sliceCount,
				aConfig.jitterEnabled,
				aConfig.jitterStrength, 
				aConfig.enableCutPlane,
				aConfig.cutPlane,
				aConfig.enableInterpolation,
				aViewSetup,
				(aConfig.integralTFEnabled ? *integralTransferFunction : *transferFunction),
				aConfig.lightPosition,
				flags
				);
			
		}
		break;
	case ctMaxIntensityProjection:
	case ctBasic:
		{
			basicRendering( 
				aConfig.camera, 
				*primaryData, 
				bbox, 
				sliceCount,
				aConfig.jitterEnabled,
				aConfig.jitterStrength, 
				aConfig.enableCutPlane,
				aConfig.cutPlane,
				aConfig.enableInterpolation,
				aConfig.lutWindow,
				aViewSetup,
				aConfig.colorTransform == ctMaxIntensityProjection,
				flags
     			);
			return;
		}
		break;
	default:
		ASSERT( false );
	}

	M4D::CheckForGLError( "OGL error : " );
}


void
VolumeRenderer::basicRendering( 
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
      			)
{
	D_PRINT("FLAGS " << aFlags);
	
	mCgEffect.setParameter( "gPrimaryImageData3D", aImage );
	mCgEffect.setParameter( "gMappedIntervalBands", aImage.GetMappedInterval() );
	
	setupView(aCamera, aViewSetup);
	setupJittering(aJitterStrength);
	setupSamplingProcess(aBoundingBox, aCamera, aSliceCount);	

	mCgEffect.setParameter("gEnableCutPlane", aEnableCutPlane );
	mCgEffect.setParameter("gCutPlane", aCutPlane );
	
	mCgEffect.setParameter("gEnableInterpolation", aEnableInterpolation );


	
	mCgEffect.setParameter("gWLWindow", aLutWindow );

	std::string techniqueName = aMIP ? "WLWindowMIP_3D" : "WLWindowBasic_3D";
	
	mCgEffect.executeTechniquePass(
			techniqueName, 
			boost::bind( &M4D::GLDrawVolumeSlices_Buffered, 
				aBoundingBox,
				aCamera,
				aSliceCount,
				mVertices,
				mIndices,
				1.0f
				) 
			); 


	M4D::CheckForGLError( "OGL error : " );
}

void
VolumeRenderer::transferFunctionRendering( 
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
	)
{
	mCgEffect.setParameter( "gPrimaryImageData3D", aImage );
	mCgEffect.setParameter( "gMappedIntervalBands", aImage.GetMappedInterval() );
	
	setupView(aCamera, aViewSetup);
	setupJittering(aJitterStrength);
	setupSamplingProcess(aBoundingBox, aCamera, aSliceCount);

	mCgEffect.setParameter("gEnableCutPlane", aEnableCutPlane );
	mCgEffect.setParameter("gCutPlane", aCutPlane );
	
	mCgEffect.setParameter("gEnableInterpolation", aEnableInterpolation );

	setupLights(aLightPosition);
	std::string techniqueName;
	if (aFlags & rf_PREINTEGRATED) {
		mCgEffect.setParameter( "gIntegralTransferFunction1D", aTransferFunction );
	} else {
		mCgEffect.setParameter( "gTransferFunction1D", aTransferFunction );
	}

	techniqueName = "TransferFunction1D";
	techniqueName += detail::gFlagsToNameSuffixes[aFlags & detail::FLAGS_TO_NAME_SUFFIXES_MASK];
	techniqueName += "_3D";
	
	mCgEffect.executeTechniquePass(
			techniqueName, 
			boost::bind( &M4D::GLDrawVolumeSlices_Buffered, 
				aBoundingBox,
				aCamera,
				aSliceCount,
				mVertices,
				mIndices,
				1.0f
				) 
			); 

	M4D::CheckForGLError( "OGL error : " );
}


}//Renderer
}//GUI
}//M4D


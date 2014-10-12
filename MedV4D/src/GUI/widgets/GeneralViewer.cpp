#include "MedV4D/GUI/widgets/GeneralViewer.h"

#include "MedV4D/GUI/utils/QtM4DTools.h"
#include "MedV4D/Common/MathTools.h"

#include "MedV4D/Imaging/ImageFactory.h"
#include "MedV4D/GUI/managers/ViewerManager.h"
#include "MedV4D/GUI/managers/ApplicationManager.h"

#include <soglu/OGLDrawing.hpp>
#include <soglu/Camera.hpp>
#include <vorgl/SliceGeneration.hpp>


namespace M4D
{
namespace GUI
{
namespace Viewer
{

void
handleCutPlane( bool aEnabled, const soglu::BoundingBox3D &aBBox, const soglu::Planef &aCutPlane, RGBAf aColor = RGBAf( 0.0f, 1.0f, 0.0f, 1.0f ) )
{
	if( !aEnabled ) return;

	glColor4fv(aColor.data());
	//GLColorVector( aColor );
	glm::fvec3 vertices[6];

	unsigned count = soglu::getPlaneVerticesInBoundingBox( aBBox, aCutPlane, vertices );
	//Render n-gon
	soglu::drawPolygon(vertices, count);
	/*glBegin( GL_LINE_LOOP );
		for( unsigned j = 0; j < count; ++j ) {
			GLVertexVector( vertices[ j ] );
		}
	glEnd();*/
}


//********************************************************************************************

GeneralViewer::GeneralViewer( QWidget *parent )
	: PredecessorType( parent )
	, _prepared( false )
{
	ViewerState * state = new ViewerState;

	state->mSliceRenderConfig.colorTransform = M4D::GUI::Renderer::ctLUTWindow;
	state->mSliceRenderConfig.plane = XY_PLANE;

	state->mVolumeRenderConfig.colorTransform = M4D::GUI::Renderer::ctMaxIntensityProjection;
	state->mVolumeRenderConfig.renderingQuality.enableJittering = true;
	state->mVolumeRenderConfig.transferFunctionOptions.enableLight = true;
	state->mVolumeRenderConfig.transferFunctionOptions.lightPosition = glm::fvec3(3000.0f, -3000.0f, 3000.0f);

	state->mVolumeRenderConfig.isoSurfaceOptions.isoSurfaceColor = glm::fvec4(0.4f, 0.4f, 1.0f, 0.5f);
	state->mVolumeRenderConfig.isoSurfaceOptions.enableLight = true;
	state->mVolumeRenderConfig.isoSurfaceOptions.lightPosition = glm::fvec3(3000.0f, -3000.0f, 3000.0f);

	state->mEnableVolumeBoundingBox = true;

	state->viewerWindow = this;

	//state->backgroundColor = QColor( 20, 10, 90);
	state->backgroundColor = QColor( 0, 0, 0, 0);

	state->availableViewTypes = 5;
	state->viewType = vt2DAlignedSlices;

	state->m2DMultiSliceGrid = Vector2u( 1, 1 );
	state->m2DMultiSliceStep = 1;

	mViewerState = BaseViewerState::Ptr( state );

	setRenderingQuality( qmNormal );

	setColorTransformType( M4D::GUI::Renderer::ctLUTWindow );

	mSliceCountForRenderingQualities = Vector4i( 90, 180, 450, 1000 );
}

void
GeneralViewer::setTiling(int aRows, int aCols)
{
	setTiling( aRows, aCols, getViewerState().m2DMultiSliceStep );
}

void
GeneralViewer::setTiling(int aRows, int aCols, int aSliceStep)
{
	ASSERT( aCols > 0 && aRows > 0 && aSliceStep > 0 );
	getViewerState().m2DMultiSliceGrid[0] = aRows;
	getViewerState().m2DMultiSliceGrid[1] = aCols;
	getViewerState().m2DMultiSliceStep = aSliceStep;
	notifyAboutSettingsChange();
	update();
}

Vector2i
GeneralViewer::getTiling() const
{
	return getViewerState().m2DMultiSliceGrid;
}

size_t
GeneralViewer::getTilingSliceStep() const
{
	return getViewerState().m2DMultiSliceStep;
}

void
GeneralViewer::setLUTWindow( float32 center, float32 width )
{
	setLUTWindow( glm::fvec2( center, width ) );
}

void
GeneralViewer::setLUTWindow(glm::fvec2 window)
{
	getViewerState().mSliceRenderConfig.lutWindow = window;
	getViewerState().mVolumeRenderConfig.densityOptions.lutWindow = window;

	notifyAboutSettingsChange();
	update();
}

glm::fvec2
GeneralViewer::getLUTWindow() const
{
	//TODO
	return getViewerState().mSliceRenderConfig.lutWindow;
}

/*void
GeneralViewer::setTransferFunctionBuffer( vorgl::TransferFunctionBuffer1D::Ptr aTFunctionBuffer )
{
	if ( !aTFunctionBuffer ) {
		_THROW_ ErrorHandling::EBadParameter();
	}
	getViewerState().mTransferFunctionBufferInfo.id = 0;
	getViewerState().mTransferFunctionBufferInfo.tfBuffer = aTFunctionBuffer;
	//getViewerState().mTFunctionBuffer = aTFunctionBuffer;

	makeCurrent();
	getViewerState().mTransferFunctionBufferInfo.tfGLBuffer = createGLTransferFunctionBuffer1D( *aTFunctionBuffer );
	//getViewerState().mTransferFunctionTexture = CreateGLTransferFunctionBuffer1D( *aTFunctionBuffer );
	doneCurrent();

	getViewerState().mSliceRenderConfig.transferFunction = getViewerState().mTransferFunctionBufferInfo.tfGLBuffer;
	getViewerState().mVolumeRenderConfig.transferFunctionOptions.transferFunction = getViewerState().mTransferFunctionBufferInfo.tfGLBuffer;

	notifyAboutSettingsChange();
	update();
}*/

void
GeneralViewer::setTransferFunctionBufferInfo( vorgl::TransferFunctionBufferInfo aTFunctionBufferInfo )
{
	makeCurrent(); //TODO - RAII
	getViewerState().mTransferFunctionBufferInfo = aTFunctionBufferInfo;

	/*getViewerState().mSliceRenderConfig.transferFunction = getViewerState().mTransferFunctionBufferInfo.tfGLBuffer;
	getViewerState().mVolumeRenderConfig.transferFunctionOptions.transferFunction = getViewerState().mTransferFunctionBufferInfo.tfGLBuffer;
	getViewerState().mVolumeRenderConfig.transferFunctionOptions.integralTransferFunction = getViewerState().mTransferFunctionBufferInfo.tfGLIntegralBuffer;*/

//	getViewerState().mSliceRenderConfig.transferFunction = getViewerState().mTransferFunctionBufferInfo;
	getViewerState().mVolumeRenderConfig.transferFunctionOptions.transferFunction = getViewerState().mTransferFunctionBufferInfo.bufferInfo;

	doneCurrent();

	notifyAboutSettingsChange();
	update();
}

vorgl::TransferFunctionBufferInfo
GeneralViewer::getTransferFunctionBufferInfo()const
{
	return getViewerState().mTransferFunctionBufferInfo;
}

/*void
GeneralViewer::setMappingBuffer( IDMappingBuffer::Ptr aMappingBuffer )
{
	getViewerState().mMappingBufferInfo.buffer = aMappingBuffer;
	getViewerState().mMappingBufferInfo.glBuffer = createGLMappingBuffer( *aMappingBuffer );
}*/

void
GeneralViewer::setCurrentSlice( int32 slice )
{
	CartesianPlanes plane = getViewerState().mSliceRenderConfig.plane;
	getViewerState().mSliceRenderConfig.currentSlice[ plane ] = max(
								min( getViewerState().getMaxSlice()[plane]-1, slice ),
								getViewerState().getMinSlice()[plane] );
	notifyAboutSettingsChange();
	update();
}

void
GeneralViewer::switchToNextPlane()
{
	getViewerState().mSliceRenderConfig.plane = NextCartesianPlane( getViewerState().mSliceRenderConfig.plane );
	zoomFit(); //TODO remove
	notifyAboutSettingsChange();
	update();
}

void
GeneralViewer::setCurrentViewPlane( CartesianPlanes aPlane )
{
	getViewerState().mSliceRenderConfig.plane = aPlane;

	zoomFit(); //TODO remove
	notifyAboutSettingsChange();
	update();
}

CartesianPlanes
GeneralViewer::getCurrentViewPlane()const
{
	return getViewerState().mSliceRenderConfig.plane;
}

int32
GeneralViewer::getCurrentSlice()const
{
	CartesianPlanes plane = getViewerState().mSliceRenderConfig.plane;
	return getViewerState().mSliceRenderConfig.currentSlice[ plane ];
}

float32
GeneralViewer::getCurrentRealSlice()const
{
	CartesianPlanes plane = getViewerState().mSliceRenderConfig.plane;
	glm::fvec3 realSlices = getViewerState().mSliceRenderConfig.getCurrentRealSlice();
	return glm::value_ptr(realSlices)[plane];
}

void
GeneralViewer::changeCurrentSlice( int32 diff )
{
	setCurrentSlice( diff + getCurrentSlice() );
}


void
GeneralViewer::setCutPlane( const soglu::Planef &aCutPlane )
{
	// TODO - handle cut planes
	/*
	getViewerState().mVolumeRenderConfig.cutPlane = aCutPlane;
	glm::fvec3 point = glm::fvec3(aCutPlane.point()[0], aCutPlane.point()[1], aCutPlane.point()[2]);
	glm::fvec3 normal = glm::fvec3(aCutPlane.normal()[0], aCutPlane.normal()[1], aCutPlane.normal()[2]);
	glm::fvec3 dir = vectorProjection(normal, point - getCameraTargetPosition());
	float offset = glm::length(dir);
	if( glm::dot(dir, normal) < 0.0f ) {
		offset *= -1.0f;
	}
	getViewerState().mVolumeRenderConfig.cutPlaneCameraTargetOffset = offset;*/

	notifyAboutSettingsChange();
	update();
}

soglu::Planef
GeneralViewer::getCutPlane()const
{
	// TODO - handle cut planes
	return soglu::Planef(); //getViewerState().mVolumeRenderConfig.cutPlane;
}

void
GeneralViewer::setCutPlaneCameraTargetOffset( float aOffset )
{
	// TODO - handle cut planes
	/*glm::fvec3 normal = getViewerState().mVolumeRenderConfig.cutPlane.normal();
	glm::fvec3 point = getCameraTargetPosition() + aOffset * glm::fvec3(normal[0], normal[1], normal[2]);
	soglu::Planef plane(glm::fvec3(point.x, point.y, point.z), normal );
	getViewerState().mVolumeRenderConfig.cutPlane = plane;
	getViewerState().mVolumeRenderConfig.cutPlaneCameraTargetOffset = aOffset;
*/
	notifyAboutSettingsChange();
	update();
}

void
GeneralViewer::updateMouseInfo( Vector3f aDataCoords )
{
	QString info = GetVoxelInfo( aDataCoords );//TODO predelat

	//LOG( "updating mouse pos info " << info.toStdString() );
	emit MouseInfoUpdate( info );
}

QString
GeneralViewer::GetVoxelInfo( Vector3f aDataCoords )
{
	//TODO improve
	/*try {
		if ( TryGetAndLockAllAvailableInputs() == 0 ) {
			return QString("NONE");
		}
	} catch (...) {
		return QString("NONE");
	}
	M4D::Imaging::AImage::ConstPtr image = M4D::Imaging::AImage::Cast( mInputDatasets[0].lock() );*/
	QString result;

	/*try {
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( image->GetElementTypeID(),
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::ConstPtr typedImage = IMAGE_TYPE::Cast( image );
			Vector3f extents = typedImage->GetElementExtents();
			// HH: previously was 'round', now is floor instead, because round was making errors during outputting correct mouse coordinates
			Vector3i coords = floor<3>( VectorMemberDivision( aDataCoords, extents ) );
			result += VectorToQString( coords );
			result += " : ";
			if( coords >= typedImage->GetMinimum() &&  coords < typedImage->GetMaximum() ) {
				result += QString::number( typedImage->GetElement( coords ) );
			} else {
				result += QString("NONE");
			}
			);

	} catch (...) {
		result = QString("NONE");
	}
	ReleaseAllInputs();*/
	return result;
}

void
GeneralViewer::getVolumeRestrictions( Vector2f &aX, Vector2f &aY, Vector2f &aZ )const
{
	//TODO test if valid values
	getViewerState().mVolumeRenderConfig.volumeRestrictions.get( aX, aY, aZ );
}

int
GeneralViewer::getColorTransformType()
{
	return getViewerState().colorTransform;//mSliceRenderConfig.colorTransform;//_renderer.GetColorTransformType();
}

QString
GeneralViewer::getColorTransformName()
{
	//TODO
	const GUI::Renderer::ColorTransformNameIDList * idList = NULL;
	switch ( getViewerState().viewType ) {
	case vt3D: idList = &getViewerState().mVolumeRenderer.GetAvailableColorTransforms();
		break;
	case vt2DAlignedSlices: idList = &getViewerState().mSliceRenderer.GetAvailableColorTransforms();
		break;
	default:
		ASSERT( false );
	}
	//ASSERT( idList && idList->size() > 0 );
	for( unsigned i = 0; i < idList->size(); ++i ) {
		if ( (*idList)[ i ].id == getViewerState().colorTransform ) {
			//return QString::fromStdWString( (*idList)[ i ].name );
			return (*idList)[ i ].name;
		}
	}

	return QString();
}

void
GeneralViewer::ReceiveMessage(
	M4D::Imaging::PipelineMessage::Ptr 			msg,
	M4D::Imaging::PipelineMessage::MessageSendStyle 	sendStyle,
	M4D::Imaging::FlowDirection				direction
)
{
	PrepareData();
	notifyAboutSettingsChange();
}

bool
GeneralViewer::isShadingEnabled() const
{
	return getViewerState().mVolumeRenderConfig.transferFunctionOptions.enableLight;
}

bool
GeneralViewer::isJitteringEnabled() const
{
	return getViewerState().mVolumeRenderConfig.renderingQuality.enableJittering;
}

bool
GeneralViewer::isIntegratedTransferFunctionEnabled() const
{
	return getViewerState().mVolumeRenderConfig.transferFunctionOptions.preintegratedTransferFunction;
}

void
GeneralViewer::enableIntegratedTransferFunction( bool aEnable )
{
	getViewerState().mVolumeRenderConfig.transferFunctionOptions.preintegratedTransferFunction = aEnable;
	notifyAboutSettingsChange();
	update();
}

bool
GeneralViewer::isVolumeRestrictionEnabled() const
{
	return getViewerState().mVolumeRenderConfig.enableVolumeRestrictions;
}

void
GeneralViewer::enableVolumeRestrictions( bool aEnable )
{
	getViewerState().mVolumeRenderConfig.enableVolumeRestrictions = aEnable;
	notifyAboutSettingsChange();
	update();
}

void
GeneralViewer::setVolumeRestrictions( const Vector2f &aX, const Vector2f &aY, const Vector2f &aZ )
{
	//TODO test if valid values
	getViewerState().mVolumeRenderConfig.volumeRestrictions = M4D::GUI::Renderer::VolumeRestrictions( aX, aY, aZ );

	notifyAboutSettingsChange();
	update();
}

void
GeneralViewer::setVolumeRestrictions( bool aEnable, const Vector2f &aX, const Vector2f &aY, const Vector2f &aZ )
{
	getViewerState().mVolumeRenderConfig.enableVolumeRestrictions = aEnable;
	getViewerState().mVolumeRenderConfig.volumeRestrictions = M4D::GUI::Renderer::VolumeRestrictions( aX, aY, aZ );

	notifyAboutSettingsChange();
	update();
}



bool
GeneralViewer::isCutPlaneEnabled() const
{
	// TODO - handle cut planes
	return false; //getViewerState().mVolumeRenderConfig.enableCutPlane;
}

void
GeneralViewer::enableCutPlane( bool aEnable )
{
	// TODO - handle cut planes
	//getViewerState().mVolumeRenderConfig.enableCutPlane = aEnable;
	notifyAboutSettingsChange();
	update();
}

bool
GeneralViewer::isInterpolationEnabled() const
{
	return getViewerState().mVolumeRenderConfig.renderingQuality.enableInterpolation;
}

void
GeneralViewer::enableInterpolation( bool aEnable )
{
	getViewerState().mVolumeRenderConfig.renderingQuality.enableInterpolation = aEnable;
	getViewerState().mSliceRenderConfig.enableInterpolation = aEnable;
	notifyAboutSettingsChange();
	update();
}


void
GeneralViewer::cameraOrbit( Vector2f aAngles )
{
	soglu::cameraYawPitchAroundPoint(getViewerState().mVolumeRenderConfig.renderingConfiguration.camera,
					glm::fvec2(aAngles[0], aAngles[1]),
					getViewerState().mVolumeRenderConfig.renderingConfiguration.camera.targetPosition()
					);
	//getViewerState().mVolumeRenderConfig.renderingConfiguration.camera.yawPitchAround( aAngles[0], aAngles[1] );
	update();
}

/*void
GeneralViewer::cameraOrbitAbsolute( Vector2f aAngles )
{
	getViewerState().mVolumeRenderConfig.renderingConfiguration.camera.yawPitchAbsolute( aAngles[0], aAngles[1] );
	update();
}*/

void
GeneralViewer::cameraDolly(float aDollyRatio)
{
	glm::fvec3 moveVector = (1.0f - aDollyRatio) * (getViewerState().mVolumeRenderConfig.renderingConfiguration.camera.targetPosition() - getViewerState().mVolumeRenderConfig.renderingConfiguration.camera.eyePosition());
	soglu::dollyCamera(getViewerState().mVolumeRenderConfig.renderingConfiguration.camera, moveVector);
	update();
}

ViewType
GeneralViewer::getViewType()const
{
	return getViewerState().viewType;
}

QStringList
GeneralViewer::getAvailableColorTransformationNames()
{
	QStringList strList;
	const GUI::Renderer::ColorTransformNameIDList * idList = NULL;
	switch ( getViewerState().viewType ) {
	case vt3D: idList = &getViewerState().mVolumeRenderer.GetAvailableColorTransforms();
		break;
	case vt2DAlignedSlices: idList = &getViewerState().mSliceRenderer.GetAvailableColorTransforms();
		break;
	default:
		ASSERT( false );
	}
	//ASSERT( idList && idList->size() > 0 );
	for( unsigned i = 0; i < idList->size(); ++i ) {
		//strList << QString::fromStdWString( (*idList)[ i ].name );
		strList << (*idList)[ i ].name;
	}

	return strList;
}

GUI::Renderer::ColorTransformNameIDList
GeneralViewer::getAvailableColorTransformations()
{
	switch ( getViewerState().viewType ) {
	case vt3D:
		return getViewerState().mVolumeRenderer.GetAvailableColorTransforms();
		break;
	case vt2DAlignedSlices:
		return getViewerState().mSliceRenderer.GetAvailableColorTransforms();
		break;
	default:
		ASSERT( false );
	}
	return GUI::Renderer::ColorTransformNameIDList();
}


void
GeneralViewer::setRenderingExtension( RenderingExtension::Ptr aRenderingExtension )
{
	mRenderingExtension = aRenderingExtension;
	notifyAboutSettingsChange();
	update();
}

void
GeneralViewer::setViewType( int aViewType )
{
	getViewerState().viewType = (ViewType)aViewType;
	//TODO
	switch ( getViewerState().viewType ) {
	case vt3D:
		setColorTransformType( GUI::Renderer::ctTransferFunction1D );
		break;
	case vt2DAlignedSlices:
		setColorTransformType( GUI::Renderer::ctLUTWindow );
		break;
	default:
		ASSERT( false );
	}


	notifyAboutSettingsChange();
	update();

	emit ViewTypeChanged( aViewType );
}

void
GeneralViewer::setColorTransformType( const QString & aColorTransformName )
{
	//TODO
	//std::wstring name = aColorTransformName.toStdWString();
	const GUI::Renderer::ColorTransformNameIDList * idList = NULL;
	switch ( getViewerState().viewType ) {
	case vt3D: idList = &getViewerState().mVolumeRenderer.GetAvailableColorTransforms();
		break;
	case vt2DAlignedSlices: idList = &getViewerState().mSliceRenderer.GetAvailableColorTransforms();
		break;
	default:
		ASSERT( false );
	}
	ASSERT( idList && idList->size() > 0 );
	for( unsigned i = 0; i < idList->size(); ++i ) {
		if( aColorTransformName == (*idList)[ i ].name ) {
			setColorTransformType( (*idList)[ i ].id );
			//D_PRINT( "Setting color transform : " << aColorTransformName.toStdString() );
			return;
		}
	}
}

void
GeneralViewer::setColorTransformType( int aColorTransform )
{
	//TODO
	getViewerState().colorTransform = aColorTransform;
	getViewerState().mSliceRenderConfig.colorTransform = aColorTransform;
	getViewerState().mVolumeRenderConfig.colorTransform = aColorTransform;

	//TODO handle in consistent way
	getViewerState().mVolumeRenderConfig.densityOptions.enableMIP = (aColorTransform == M4D::GUI::Renderer::ctMaxIntensityProjection);

	notifyAboutSettingsChange();
	update();

	emit ColorTransformTypeChanged( aColorTransform );
}

void
GeneralViewer::fineRender()
{
	//_renderer.FineRender();
	update();
}

void
GeneralViewer::enableShading( bool aEnable )
{
	getViewerState().mVolumeRenderConfig.transferFunctionOptions.enableLight = aEnable;
	notifyAboutSettingsChange();
	update();
}

void
GeneralViewer::enableJittering( bool aEnable )
{
	getViewerState().mVolumeRenderConfig.renderingQuality.enableJittering = aEnable;
	notifyAboutSettingsChange();
	update();
}


void
GeneralViewer::setJitterStrength( float aValue )
{
	getViewerState().mVolumeRenderConfig.renderingQuality.jitterStrength = abs(aValue);
	notifyAboutSettingsChange();
	update();
}


void
GeneralViewer::resetView()
{
	glm::fvec3 pos = getViewerState().mVolumeRenderConfig.renderingConfiguration.camera.targetPosition();
	pos[1] += -550;
	getViewerState().mVolumeRenderConfig.renderingConfiguration.camera.setEyePosition( pos, glm::fvec3( 0.0f, 0.0f, 1.0f ) );

	update();
}

QualityMode
GeneralViewer::getRenderingQuality()
{
	return getViewerState().mQualityMode;
}

void
GeneralViewer::setSliceCountForRenderingQualities( int aLow, int aNormal, int aHigh, int aFinest )
{
	mSliceCountForRenderingQualities[qmLow] = aLow;
	mSliceCountForRenderingQualities[qmNormal] = aNormal;
	mSliceCountForRenderingQualities[qmHigh] = aHigh;
	mSliceCountForRenderingQualities[qmFinest] = aFinest;
}


void
GeneralViewer::setRenderingQuality( int aQualityMode )
{
	//TODO check
	getViewerState().mQualityMode = static_cast< QualityMode >( aQualityMode );

	getViewerState().mVolumeRenderConfig.renderingQuality.sliceCount = mSliceCountForRenderingQualities[aQualityMode];
	notifyAboutSettingsChange();
	update();
}

bool
GeneralViewer::isBoundingBoxEnabled()const
{
	return getViewerState().mEnableVolumeBoundingBox;
}

void
GeneralViewer::enableBoundingBox( bool aEnable )
{
	getViewerState().mEnableVolumeBoundingBox = aEnable;
	notifyAboutSettingsChange();
	update();
}

//********************************************************************************

void
GeneralViewer::initializeRenderingEnvironment()
{
	getViewerState().mSliceRenderer.initialize();
	getViewerState().mVolumeRenderer.initialize();

	boost::filesystem::path dataDirName = GET_SETTINGS_NODEFAULT( "application.data_directory", std::string );
	/*getViewerState().mSceneSlicingCgEffect.initialize( dataDirName / "shaders" / "SceneSlicing.cgfx" );
	getViewerState().mBasicCgEffect.initialize( dataDirName / "shaders" / "BasicShader.cgfx" );*/

	getViewerState().basicShaderProgram = soglu::createGLSLProgramFromVertexAndFragmentShader(
		dataDirName / "shaders" / "basic_vertex.glsl",
		dataDirName / "shaders" / "basic_fragment.glsl");
}

bool
GeneralViewer::preparedForRendering()
{
	if( !IsDataPrepared() /*&& !PrepareData()*/ ) {
		return false;
	}
	return true;
}

void
GeneralViewer::prepareForRenderingStep()
{
	switch ( getViewerState().viewType ) {
	case vt3D:
		{
			getViewerState().mVolumeRenderConfig.renderingConfiguration.camera.setAspectRatio( getViewerState().aspectRatio );
			//Set viewing parameters
			getViewerState().glViewSetup = getViewSetupFromCamera( getViewerState().mVolumeRenderConfig.renderingConfiguration.camera );
			getViewerState().glViewSetup.viewport = glm::ivec4( 0, 0, width(), height() );
			getViewerState().mVolumeRenderConfig.renderingConfiguration.viewSetup = getViewerState().glViewSetup;
		}
		break;
	case vt2DAlignedSlices:
		{
			int subVPortW = width() / getViewerState().m2DMultiSliceGrid[1];
			int subVPortH = height() / getViewerState().m2DMultiSliceGrid[0];
			CartesianPlanes plane = getViewerState().mSliceRenderConfig.plane;
			Vector2f size = VectorPurgeDimension( getViewerState().getRealSize(), plane );
			float zoom = M4D::min( float(subVPortW) / size[0], float(subVPortH) / size[1] );

			getViewerState().mSliceRenderConfig.camera.setWindow( subVPortW / zoom, subVPortH / zoom );
			getViewerState().mSliceRenderConfig.camera.setTargetPosition( glm::fvec3(getViewerState().getRealCenter()[0], getViewerState().getRealCenter()[1], getViewerState().getRealCenter()[2]) );
			getViewerState().mSliceRenderConfig.sliceCenter = toGLM(getViewerState().getRealCenter());
			getViewerState().mSliceRenderConfig.sliceCenter[plane] = float32(getViewerState().mSliceRenderConfig.currentSlice[ plane ]+0.5f) * getViewerState().mSliceRenderConfig.primaryImageData.lock()->getExtents().elementExtents[plane];
			glm::vec3 eye = getViewerState().mSliceRenderConfig.camera.targetPosition();
			glm::vec3 up;
			switch ( plane ) {
			case YZ_PLANE:
				up = glm::vec3( 0.0f, 0.0f, 1.0f );
				eye[0] =  + 500.0f;
				break;
			case XZ_PLANE:
				up = glm::vec3( 0.0f, 0.0f, 1.0f );
				eye[1] = 500.0f;
				break;
			case XY_PLANE:
				up = glm::vec3( 0.0f, 1.0f, 0.0f );
				eye[2] = 500.0f;
				break;
			default:
				ASSERT( false );
			}
			getViewerState().mSliceRenderConfig.camera.setEyePosition(eye, up);
			getViewerState().mSliceRenderConfig.sliceNormal = glm::normalize(getViewerState().mSliceRenderConfig.camera.targetDirection());
			getViewerState().glViewSetup = soglu::getViewSetupFromOrthoCamera( getViewerState().mSliceRenderConfig.camera );
			getViewerState().glViewSetup.viewport = glm::ivec4( 0, 0, subVPortW, subVPortH );

		}
		break;
	default:
		ASSERT( false );
	}
}

void
GeneralViewer::render()
{
	switch ( getViewerState().viewType ) {
	case vt3D:
		{
			GL_CHECKED_CALL(glEnable(GL_BLEND));
			GL_CHECKED_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
			int vertexLocation = getViewerState().basicShaderProgram.getAttributeLocation("vertex");
			glViewport(0, 0, width(), height());

			getViewerState().basicShaderProgram.setUniformByName("gViewSetup", getViewerState().glViewSetup);

			soglu::ExtentsRecord<3> extents = getViewerState().mVolumeRenderConfig.primaryImageData.lock()->getExtents();
			soglu::BoundingBox3D bbox(extents.realMaximum, extents.realMinimum);
			GL_CHECKED_CALL(glEnable(GL_DEPTH_TEST));
			//GL_CHECKED_CALL( glDisable( GL_LIGHTING ) );
			/*getViewerState().mBasicCgEffect.setParameter( "gViewSetup", getViewerState().glViewSetup );*/
			if ( getViewerState().mEnableVolumeBoundingBox ) {
				getViewerState().basicShaderProgram.setUniformByName("fragmentColor", glm::fvec4(1, 0, 0, 1));
				getViewerState().basicShaderProgram.use([&bbox, vertexLocation]()
				{
					//soglu::drawVertexIndexBuffers(soglu::generateBoundingBoxBuffers(bbox), GL_TRIANGLE_STRIP, vertexLocation);
					soglu::drawVertexIndexBuffers(soglu::generateBoundingBoxBuffersWireframe(bbox), GL_LINE_STRIP, vertexLocation);
				});
			}
			//Draw cut plane if enabled TODO - set color
			/*getViewerState().mBasicCgEffect.executeTechniquePass(
				"Basic", boost::bind( &M4D::GUI::Viewer::handleCutPlane, getViewerState().mVolumeRenderConfig.enableCutPlane, bbox, getViewerState().mVolumeRenderConfig.cutPlane, M4D::RGBAf( 0.0f, 1.0f, 0.0f, 1.0f ) )
				);*/

			//LOG( getViewerState().glViewSetup );
			/*if ( mRenderingExtension && (vt3D | mRenderingExtension->getAvailableViewTypes()) ) {
				getViewerState().mBasicCgEffect.executeTechniquePass( "Basic", boost::bind( &M4D::GUI::Viewer::RenderingExtension::preRender3D, mRenderingExtension ) );
				//mRenderingExtension->preRender3D();
			}*/

			try {
				getViewerState().mVolumeRenderConfig.renderingConfiguration.depthBuffer = mFrameBufferObject.mDepthAttachment;
				getViewerState().mVolumeRenderConfig.renderingConfiguration.windowSize = glm::ivec2(width(), height());
				//getViewerState().mVolumeRenderConfig.renderingConfiguration.viewSetup = getViewerState().glViewSetup;
				//getViewerState().mVolumeRenderConfig.renderingConfiguration.camera = getViewerState().camera;

				getViewerState().mVolumeRenderer.Render( getViewerState().mVolumeRenderConfig, getViewerState().glViewSetup );
			} catch( std::exception &e ) {
				LOG( e.what() );
			}

			GL_CHECKED_CALL( glClear( GL_DEPTH_BUFFER_BIT ) );//TODO disable depth storing during volume rendering
			/*if ( mRenderingExtension && (vt3D | mRenderingExtension->getAvailableViewTypes()) ) {
				getViewerState().mBasicCgEffect.executeTechniquePass( "Basic", boost::bind( &M4D::GUI::Viewer::RenderingExtension::postRender3D, mRenderingExtension ) );
				//mRenderingExtension->postRender3D();
			}*/
		}
		break;
	case vt2DAlignedSlices:
		{
			ASSERT( getViewerState().m2DMultiSliceGrid[0] > 0 && getViewerState().m2DMultiSliceGrid[1] > 0 );
			int subVPortW = width() / getViewerState().m2DMultiSliceGrid[1];
			int subVPortH = height() / getViewerState().m2DMultiSliceGrid[0];

			//size_t sliceOffset = 0;
			Renderer::SliceRenderer::RenderingConfiguration config = getViewerState().mSliceRenderConfig;
			for (int j = 0; j < getViewerState().m2DMultiSliceGrid[0]; ++j ) {
				for (int i = 0; i < getViewerState().m2DMultiSliceGrid[1]; ++i ) {

					GL_CHECKED_CALL( glViewport( i * subVPortW, j * subVPortH, subVPortW, subVPortH ) );
					try {
						/*int vertexLocation = getViewerState().mBasicShaderProgram.getAttributeLocation("vertex");
						getViewerState().mBasicShaderProgram.setUniformByName("gViewSetup", getViewerState().glViewSetup);

						soglu::ExtentsRecord<3> extents = getViewerState().mVolumeRenderConfig.primaryImageData.lock()->getExtents();
						soglu::BoundingBox3D bbox(extents.realMaximum, extents.realMinimum);
							getViewerState().mBasicShaderProgram.use([&, this]()
							{
								//soglu::drawVertexIndexBuffers(soglu::generateBoundingBoxBuffers(bbox), GL_LINE_STRIP,	vertexLocation);
								soglu::drawVertexBuffer(
									vorgl::generateVolumeSlice(extents.realMinimum, extents.realMaximum, 0.5f,
									//this->getViewerState().mSliceRenderConfig.currentSlice,
									(soglu::CartesianPlanes)this->getViewerState().mSliceRenderConfig.plane),
									GL_LINE_LOOP,
								vertexLocation
								);
							});*/

						getViewerState().mSliceRenderer.render(config, getViewerState().glViewSetup);
					}catch( std::exception &e ) {
						LOG( e.what() );
					}

					glClear( GL_DEPTH_BUFFER_BIT );

					if ( mRenderingExtension && (vt2DAlignedSlices | mRenderingExtension->getAvailableViewTypes()) ) {
						CartesianPlanes plane = config.plane;
						Vector3f realSlices = fromGLM(config.getCurrentRealSlice());
						Vector3f hextents = 0.5f * getViewerState().getMinimalElementExtents();
						/*getViewerState().mSceneSlicingCgEffect.setParameter( "gPlaneNormal", getViewerState().mSliceRenderConfig.sliceNormal );
						getViewerState().mSceneSlicingCgEffect.setParameter( "gPlanePoint", getViewerState().mSliceRenderConfig.sliceCenter );
						getViewerState().mSceneSlicingCgEffect.setParameter( "gPlaneWidth", 2*hextents[plane] );
						getViewerState().mSceneSlicingCgEffect.setParameter( "gViewSetup", getViewerState().glViewSetup );
						getViewerState().mSceneSlicingCgEffect.executeTechniquePass(
									"SceneSlicing",
									boost::bind( &M4D::GUI::Viewer::RenderingExtension::render2DAlignedSlices,
									mRenderingExtension,
									config.currentSlice[ config.plane ],
									Vector2f( realSlices[plane] - hextents[plane], realSlices[plane] + hextents[plane] ),
									config.plane
									) );*/
						/*mRenderingExtension->render2DAlignedSlices( config.currentSlice[ config.plane ],
								Vector2f( realSlices[plane] - hextents[plane], realSlices[plane] + hextents[plane] ),
								config.plane
								);*/
					}
					//sliceOffset += getViewerState().m2DMultiSliceStep;
					config.currentSlice[ config.plane ] += getViewerState().m2DMultiSliceStep;
				}
			}
		}
		break;
	default:
		ASSERT( false );
	}
}

void
GeneralViewer::finalizeAfterRenderingStep()
{

}

MouseEventInfo
GeneralViewer::getMouseEventInfo( QMouseEvent * event )
{
	switch ( getViewerState().viewType ) {
	case vt3D:
		{
			//throw "TODO";
			glm::fvec3 direction;
			//LOG( mViewerState->glViewSetup );
			try{
				direction = soglu::getDirectionFromScreenCoordinatesAndCameraPosition(
					glm::fvec2( event->localPos().x(), mViewerState->glViewSetup.viewport[3] - event->localPos().y() ),
					mViewerState->glViewSetup,
					getCameraPosition()
					);
			} catch (...){
				LOG( "Unprojecting of screen coordinates failed" );
			}

			return MouseEventInfo( mViewerState->glViewSetup, event, vt3D, getCameraPosition(), direction );
		}
		break;
	case vt2DAlignedSlices:
		{
			int subVPortW = width() / getViewerState().m2DMultiSliceGrid[1];
			int subVPortH = height() / getViewerState().m2DMultiSliceGrid[0];
			glm::fvec3 eye = getViewerState().mSliceRenderConfig.camera.eyePosition();
			glm::fvec3 target = getViewerState().mSliceRenderConfig.camera.targetPosition();
			glm::fvec3 dir = target - eye;
			dir = glm::normalize(dir);

			int x = event->pos().x() / subVPortW;
			int y = event->pos().y() / subVPortH;
			Vector3f slicePoint = fromGLM(getViewerState().mSliceRenderConfig.sliceCenter);
			glm::dvec3 pom = soglu::getPointFromScreenCoordinates(
						glm::fvec2(event->localPos().x() - subVPortW * x, mViewerState->glViewSetup.viewport[3] - event->localPos().y() + subVPortH * y),
						mViewerState->glViewSetup
						);
			//LOG( event->localPos().x() << ";  " << event->localPos().y() );
			Vector3f intersection;
			/*IntersectionResult res =*/ AxisPlaneIntersection(
					Vector3f(pom.x, pom.y, pom.z),
					Vector3f(dir.x, dir.y, dir.z),
					slicePoint,
					Vector3f(dir.x, dir.y, dir.z),
					intersection
					);
			//LOG( "Intersection res = " << res << "; " << intersection );

			/*Vector2f pos = GetRealCoordinatesFromScreen(
				Vector2f( event->localPos().x(), event->localPos().y() ),
				getViewerState().mWindowSize,
				getViewerState().mSliceRenderConfig.viewConfig
				);
			float32 realSlice = getCurrentRealSlice();*/
			//Vector3f position = VectorInsertDimension( pos, realSlice, getCurrentViewPlane() );
			glm::fvec3 position = toGLM(intersection);
			return MouseEventInfo( mViewerState->glViewSetup, event, vt2DAlignedSlices, position );
		}
		break;
	default:
		ASSERT( false );
	}
	return MouseEventInfo( mViewerState->glViewSetup, NULL, vt3D ); //Shouldn't reach this
}

//***********************************************************************************

void
GeneralViewer::notifyAboutSettingsChange()
{
	emit settingsChanged();
	if ( mSelected ) { //TODO make differently - not dependent on ViewerManager
		ViewerManager::getInstance()->notifyAboutChangedViewerSettings();
	}
}

bool
GeneralViewer::IsDataPrepared()
{
	return _prepared;
}

bool
GeneralViewer::PrepareData()
{
	D_BLOCK_COMMENT( "Entering PrepareData() method", "Leaving PrepareData() method" );
	/*try {
		if( TryGetAndLockAllAvailableInputs() == 0 ){
			return false;
		};
	} catch (std::exception &) {
		return false;
	}*/
	if (!mData) {
		return false;
	}

	if (mData->primaryImage()) {
		getViewerState().mPrimaryImageExtents = mData->primaryImage()->GetImageExtentsRecord();
		getViewerState().mPrimaryImageTexture = OpenGLManager::getInstance()->getTextureFromImage(*(mData->primaryImage()));
	} else {
		return false;
	}
	if (mData->secondaryImage()) {;
		getViewerState().mSecondaryImageExtents = mData->secondaryImage()->GetImageExtentsRecord();
		getViewerState().mSecondaryImageTexture = OpenGLManager::getInstance()->getTextureFromImage(*(mData->secondaryImage()));
	} else {
		getViewerState().mSecondaryImageTexture.reset();
	}

	//ReleaseAllInputs();

	getViewerState().mSliceRenderConfig.currentSlice = toGLM(getViewerState().mPrimaryImageExtents.minimum);
	getViewerState().mSliceRenderConfig.primaryImageData = soglu::GLTextureGetDimensionedInterfaceWPtr<3>( getViewerState().mPrimaryImageTexture );
	getViewerState().mVolumeRenderConfig.primaryImageData = soglu::GLTextureGetDimensionedInterfaceWPtr<3>( getViewerState().mPrimaryImageTexture );
	if ( getViewerState().mSecondaryImageTexture.lock() ) {
		getViewerState().mSliceRenderConfig.secondaryImageData = soglu::GLTextureGetDimensionedInterfaceWPtr<3>( getViewerState().mSecondaryImageTexture );
		getViewerState().mVolumeRenderConfig.secondaryImageData = soglu::GLTextureGetDimensionedInterfaceWPtr<3>( getViewerState().mSecondaryImageTexture );
		D_PRINT( "Secondary image prepared" );
	} else {
		getViewerState().mSliceRenderConfig.secondaryImageData.reset();
		getViewerState().mVolumeRenderConfig.secondaryImageData.reset();
	}

	Vector3f tmp = getViewerState().getRealCenter();
	getViewerState().mVolumeRenderConfig.renderingConfiguration.camera.setTargetPosition(glm::fvec3(tmp[0], tmp[1], tmp[2]));
	getViewerState().mVolumeRenderConfig.renderingConfiguration.camera.setFieldOfView( 45.0f );
	getViewerState().mVolumeRenderConfig.renderingConfiguration.camera.moveTo(glm::fvec3( 0.0f, 0.0f, 750.0f ));
	resetView();

	_prepared = true;
	zoomFit(); //TODO remove
	return true;
}

void
GeneralViewer::zoomFit( ZoomType zoomType )
{
	/*getViewerState().mSliceRenderConfig.viewConfig = GetOptimalViewConfiguration(//TODO
			VectorPurgeDimension( getViewerState()._regionRealMin, getViewerState().mSliceRenderConfig.plane ),
			VectorPurgeDimension( getViewerState()._regionRealMax, getViewerState().mSliceRenderConfig.plane ),
			Vector< uint32, 2 >( (uint32)width(), (uint32)height() ),
			zoomType );*/
	emit settingsChanged();
	update();
}

void
GeneralViewer::setZoom( ZoomType zoomType )
{
	zoomFit( zoomType );
}

void
GeneralViewer::setZoom( float aZoom )
{
	/*const Vector2f regionMin = VectorPurgeDimension( getViewerState()._regionRealMin, getViewerState().mSliceRenderConfig.plane );//TODO
	const Vector2f regionMax = VectorPurgeDimension( getViewerState()._regionRealMax, getViewerState().mSliceRenderConfig.plane );
	getViewerState().mSliceRenderConfig.viewConfig = ViewConfiguration2D( regionMin + (0.5f * (regionMax - regionMin)), aZoom );*/
	emit settingsChanged();
	update();
}

QStringList
GeneralViewer::getPredefinedZoomValueNames()const
{
	QStringList zoom;
	zoom << tr( "50%" );
	zoom << tr( "75%" );
	zoom << tr( "100%" );
	zoom << tr( "150%" );
	zoom << tr( "200%" );
	zoom << tr( "400%" );
	zoom << tr( "Best Fit" );
	zoom << tr( "Fit Width" );
	zoom << tr( "Fit Height" );
	return zoom;
}

QString
GeneralViewer::getZoomValueName()const
{
	//LOG( getViewerState().mSliceRenderConfig.viewConfig.zoom );
	return QString::number( getViewerState().mSliceRenderConfig.viewConfig.zoom * 100.0 ) +"%";
	//return QString();
}

void GeneralViewer::setInputData(ViewerInputData::ConstPtr aData)
{
	mData = aData;
	PrepareData();
	notifyAboutSettingsChange();
}

void
GeneralViewer::setZoom( const QString &aZoom )
{
	QRegExp percentString( "\\s*(\\d+\\.?\\d*)\\s*%?\\s*" );
	if( percentString.exactMatch( aZoom ) ) {
		QString text = percentString.cap(1);
		bool ok= false;
		float zoom = static_cast< float >( text.toDouble(&ok) );
		LOG( text.toLocal8Bit().data() << "   " << zoom );
		if( ok ) {
			setZoom( zoom * 0.01f );
			return;
		}
	}
	//LOG( aZoom.toLocal8Bit().data() << "   " << percentString.exactMatch( aZoom ) );
}


void
GeneralViewer::reloadShaders()
{
	makeCurrent();
	// TODO exception safety
	getViewerState().mVolumeRenderer.reloadShaders();
	getViewerState().mSliceRenderer.reloadShaders();
	doneCurrent();
}

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/


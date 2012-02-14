#include "MedV4D/GUI/widgets/GeneralViewer.h"
#include "MedV4D/GUI/utils/OGLDrawing.h"
#include "MedV4D/GUI/utils/QtM4DTools.h"
#include "MedV4D/Common/MathTools.h"

#include "MedV4D/Imaging/ImageFactory.h"
//#include "MedV4D/GUI/utils/CameraManipulator.h"
#include "MedV4D/GUI/managers/ViewerManager.h"
#include "MedV4D/GUI/managers/ApplicationManager.h"



namespace M4D
{
namespace GUI
{
namespace Viewer
{
	
void
handleCutPlane( bool aEnabled, const M4D::BoundingBox3D &aBBox, const Planef &aCutPlane, RGBAf aColor = RGBAf( 0.0f, 1.0f, 0.0f, 1.0f ) )
{
	if( !aEnabled ) return;
	
	GLColorVector( aColor );
	Vector< float, 3> vertices[6];
	
	unsigned count = M4D::GetPlaneVerticesInBoundingBox( aBBox, aCutPlane, vertices );
	//Render n-gon
	glBegin( GL_LINE_LOOP );
		for( unsigned j = 0; j < count; ++j ) {
			GLVertexVector( vertices[ j ] );
		}
	glEnd();
}
	

//********************************************************************************************

GeneralViewer::GeneralViewer( QWidget *parent ): PredecessorType( parent ), _prepared( false )
{
	ViewerState * state = new ViewerState;

	state->mSliceRenderConfig.colorTransform = M4D::GUI::Renderer::ctLUTWindow;
	state->mSliceRenderConfig.plane = XY_PLANE;

	state->mVolumeRenderConfig.colorTransform = M4D::GUI::Renderer::ctMaxIntensityProjection;
	state->mVolumeRenderConfig.shadingEnabled = true;
	state->mVolumeRenderConfig.jitterEnabled = true;
	state->mVolumeRenderConfig.lightPosition = Vector3f( 3000.0f, -3000.0f, 3000.0f );

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
GeneralViewer::setTiling( unsigned aRows, unsigned aCols )
{
	setTiling( aRows, aCols, getViewerState().m2DMultiSliceStep );
}

void
GeneralViewer::setTiling( unsigned aRows, unsigned aCols, unsigned aSliceStep )
{
	ASSERT( aCols > 0 && aRows > 0 && aSliceStep > 0 );
	getViewerState().m2DMultiSliceGrid[0] = aRows;
	getViewerState().m2DMultiSliceGrid[1] = aCols;
	getViewerState().m2DMultiSliceStep = aSliceStep;
	notifyAboutSettingsChange();
	update();
}

Vector2u
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
	setLUTWindow( Vector2f( center, width ) );
}

void
GeneralViewer::setLUTWindow( Vector2f window )
{
	getViewerState().mSliceRenderConfig.lutWindow = window;
	getViewerState().mVolumeRenderConfig.lutWindow = window;

	notifyAboutSettingsChange();
	update();
}

Vector2f
GeneralViewer::getLUTWindow() const
{
	//TODO
	return getViewerState().mSliceRenderConfig.lutWindow;
}

void
GeneralViewer::setTransferFunctionBuffer( TransferFunctionBuffer1D::Ptr aTFunctionBuffer )
{
	if ( !aTFunctionBuffer ) {
		_THROW_ ErrorHandling::EBadParameter();
	}
	getViewerState().mTransferFunctionBufferInfo.id = 0;
	getViewerState().mTransferFunctionBufferInfo.tfBuffer = aTFunctionBuffer;
	//getViewerState().mTFunctionBuffer = aTFunctionBuffer;
	
	makeCurrent();
	getViewerState().mTransferFunctionBufferInfo.tfGLBuffer = CreateGLTransferFunctionBuffer1D( *aTFunctionBuffer );	
	//getViewerState().mTransferFunctionTexture = CreateGLTransferFunctionBuffer1D( *aTFunctionBuffer );
	doneCurrent();

	getViewerState().mSliceRenderConfig.transferFunction = getViewerState().mTransferFunctionBufferInfo.tfGLBuffer.get();
	getViewerState().mVolumeRenderConfig.transferFunction = getViewerState().mTransferFunctionBufferInfo.tfGLBuffer.get();

	notifyAboutSettingsChange();
	update();
}

void
GeneralViewer::setTransferFunctionBufferInfo( TransferFunctionBufferInfo aTFunctionBufferInfo )
{
	getViewerState().mTransferFunctionBufferInfo = aTFunctionBufferInfo;

	getViewerState().mSliceRenderConfig.transferFunction = getViewerState().mTransferFunctionBufferInfo.tfGLBuffer.get();
	getViewerState().mVolumeRenderConfig.transferFunction = getViewerState().mTransferFunctionBufferInfo.tfGLBuffer.get();
	getViewerState().mVolumeRenderConfig.integralTransferFunction = getViewerState().mTransferFunctionBufferInfo.tfGLIntegralBuffer.get();

	notifyAboutSettingsChange();
	update();
}

TransferFunctionBufferInfo
GeneralViewer::getTransferFunctionBufferInfo()const
{
	return getViewerState().mTransferFunctionBufferInfo;
}

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
	Vector3f realSlices = getViewerState().mSliceRenderConfig.getCurrentRealSlice();
	return realSlices[plane];
}

void
GeneralViewer::changeCurrentSlice( int32 diff )
{
	setCurrentSlice( diff + getCurrentSlice() );
}


void
GeneralViewer::setCutPlane( const Planef &aCutPlane )
{
	getViewerState().mVolumeRenderConfig.cutPlane = aCutPlane;
	Vector3f point = aCutPlane.point();
	Vector3f dir = VectorProjection(aCutPlane.normal(), point - getCameraTargetPosition());
	float offset = VectorSize( dir );
	if( dir * aCutPlane.normal() < 0.0f ) {
		offset *= -1.0f;
	}
	getViewerState().mVolumeRenderConfig.cutPlaneCameraTargetOffset = offset;

	notifyAboutSettingsChange();
	update();
}

Planef
GeneralViewer::getCutPlane()const
{
	return getViewerState().mVolumeRenderConfig.cutPlane;
}

void
GeneralViewer::setCutPlaneCameraTargetOffset( float aOffset )
{
	Vector3f normal = getViewerState().mVolumeRenderConfig.cutPlane.normal();
	Planef plane( getCameraTargetPosition() + aOffset * normal, normal );
	getViewerState().mVolumeRenderConfig.cutPlane = plane;
	getViewerState().mVolumeRenderConfig.cutPlaneCameraTargetOffset = aOffset;

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
	try {
		TryGetAndLockAllInputs();
	} catch (...) {
		return QString("NONE");
	}
	M4D::Imaging::AImage::ConstPtr image = M4D::Imaging::AImage::Cast( mInputDatasets[0] );
	QString result;

	try {
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
	ReleaseAllInputs();
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
	ASSERT( idList && idList->size() > 0 );
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
	return getViewerState().mVolumeRenderConfig.shadingEnabled;
}

bool
GeneralViewer::isJitteringEnabled() const
{
	return getViewerState().mVolumeRenderConfig.jitterEnabled;
}

bool
GeneralViewer::isIntegratedTransferFunctionEnabled() const
{
	return getViewerState().mVolumeRenderConfig.integralTFEnabled;
}

void
GeneralViewer::enableIntegratedTransferFunction( bool aEnable )
{
	getViewerState().mVolumeRenderConfig.integralTFEnabled = aEnable;
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
	return getViewerState().mVolumeRenderConfig.enableCutPlane;
}

void
GeneralViewer::enableCutPlane( bool aEnable )
{
	getViewerState().mVolumeRenderConfig.enableCutPlane = aEnable;
	notifyAboutSettingsChange();
	update();
}

bool
GeneralViewer::isInterpolationEnabled() const
{
	return getViewerState().mVolumeRenderConfig.enableInterpolation;
}

void
GeneralViewer::enableInterpolation( bool aEnable )
{
	getViewerState().mVolumeRenderConfig.enableInterpolation = aEnable;
	getViewerState().mSliceRenderConfig.enableInterpolation = aEnable;
	notifyAboutSettingsChange();
	update();
}


void
GeneralViewer::cameraOrbit( Vector2f aAngles )
{
	getViewerState().mVolumeRenderConfig.camera.YawPitchAround( aAngles[0], aAngles[1] );
	update();
}

void
GeneralViewer::cameraOrbitAbsolute( Vector2f aAngles )
{
	getViewerState().mVolumeRenderConfig.camera.YawPitchAbsolute( aAngles[0], aAngles[1] );
	update();
}

void
GeneralViewer::cameraDolly( float aDollyRatio )
{
	DollyCamera( getViewerState().mVolumeRenderConfig.camera, aDollyRatio );
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
	ASSERT( idList && idList->size() > 0 );
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
	getViewerState().mVolumeRenderConfig.shadingEnabled = aEnable;
	notifyAboutSettingsChange();
	update();
}

void
GeneralViewer::enableJittering( bool aEnable )
{
	getViewerState().mVolumeRenderConfig.jitterEnabled = aEnable;
	notifyAboutSettingsChange();
	update();
}


void
GeneralViewer::setJitterStrength( float aValue )
{
	getViewerState().mVolumeRenderConfig.jitterStrength = abs(aValue);
	notifyAboutSettingsChange();
	update();
}


void
GeneralViewer::resetView()
{
	Vector3f pos = getViewerState().mVolumeRenderConfig.camera.GetTargetPosition();
	pos[1] += -550;
	getViewerState().mVolumeRenderConfig.camera.SetEyePosition( pos, Vector3f( 0.0f, 0.0f, 1.0f ) );
	
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

	getViewerState().mVolumeRenderConfig.sampleCount = mSliceCountForRenderingQualities[aQualityMode];

	/*switch ( getViewerState().mQualityMode ) {
	case qmLow:
		getViewerState().mVolumeRenderConfig.sampleCount = GET_SETTINGS( "gui.viewer.volume_rendering.sample_count_low_quality", int, 90 );
		break;
	case qmNormal:
		getViewerState().mVolumeRenderConfig.sampleCount = GET_SETTINGS( "gui.viewer.volume_rendering.sample_count_normal_quality", int, 180 );
		break;
	case qmHigh:
		getViewerState().mVolumeRenderConfig.sampleCount = GET_SETTINGS( "gui.viewer.volume_rendering.sample_count_high_quality", int, 450 );
		break;
	case qmFinest:
		getViewerState().mVolumeRenderConfig.sampleCount = GET_SETTINGS( "gui.viewer.volume_rendering.sample_count_finest_quality", int, 1000 );
		break;
	}*/
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
	getViewerState().mSliceRenderer.Initialize();
	getViewerState().mVolumeRenderer.Initialize();
	
	getViewerState().mSceneSlicingCgEffect.Initialize( "data/shaders/SceneSlicing.cgfx" );
	getViewerState().mBasicCgEffect.Initialize( "data/shaders/BasicShader.cgfx" );
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
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	switch ( getViewerState().viewType ) {
	case vt3D:
		{
			getViewerState().mVolumeRenderConfig.camera.SetAspectRatio( getViewerState().aspectRatio );
			//Set viewing parameters
			/*SetViewAccordingToCamera( getViewerState().mVolumeRenderConfig.camera );
			GLViewSetup tmpSetup;
			getCurrentGLSetup( tmpSetup );*/
			getViewerState().glViewSetup = getViewSetupFromCamera( getViewerState().mVolumeRenderConfig.camera );
			//LOG( "******************************\n" << tmpSetup );
			//LOG( "------------------------------\n" << getViewerState().glViewSetup );
		}
		break;
	case vt2DAlignedSlices:
		{
			//SetToViewConfiguration2D( getViewerState().mSliceRenderConfig.viewConfig );
			
			int subVPortW = width() / getViewerState().m2DMultiSliceGrid[1];
			int subVPortH = height() / getViewerState().m2DMultiSliceGrid[0];
			CartesianPlanes plane = getViewerState().mSliceRenderConfig.plane;
			/*Vector2f regMin = VectorPurgeDimension( getViewerState()._regionRealMin, plane ); 
			Vector2f regMax = VectorPurgeDimension( getViewerState()._regionRealMax, plane );*/
			Vector2f size = VectorPurgeDimension( getViewerState().getRealSize(), plane );
			float zoom = M4D::min( float(subVPortW) / size[0], float(subVPortH) / size[1] );
			
			getViewerState().mSliceRenderConfig.camera.SetWindow( subVPortW / zoom, subVPortH / zoom );
			getViewerState().mSliceRenderConfig.camera.SetTargetPosition( getViewerState().getRealCenter() );
			getViewerState().mSliceRenderConfig.sliceCenter = getViewerState().getRealCenter();
			getViewerState().mSliceRenderConfig.sliceCenter[plane] = float32(getViewerState().mSliceRenderConfig.currentSlice[ plane ]+0.5f) * getViewerState().mSliceRenderConfig.primaryImageData->GetElementExtents()[plane];
			Vector3f eye = getViewerState().mSliceRenderConfig.camera.GetTargetPosition();
			Vector3f up;
			switch ( plane ) {
			case YZ_PLANE:
				up = Vector3f( 0.0f, 0.0f, 1.0f );
				eye[0] =  + 500.0f;
				break;
			case XZ_PLANE:
				up = Vector3f( 0.0f, 0.0f, 1.0f );
				eye[1] = 500.0f;
				break;
			case XY_PLANE:
				up = Vector3f( 0.0f, 1.0f, 0.0f );
				eye[2] = 500.0f;
				break;
			default:
				ASSERT( false );
			}
			getViewerState().mSliceRenderConfig.camera.SetEyePosition( eye, up );
			
			getViewerState().mSliceRenderConfig.sliceNormal = getViewerState().mSliceRenderConfig.camera.GetTargetDirection();
			VectorNormalization( getViewerState().mSliceRenderConfig.sliceNormal );
			getViewerState().glViewSetup = getViewSetupFromOrthoCamera( getViewerState().mSliceRenderConfig.camera );
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
			glViewport(0, 0, width(), height());
			
			M4D::BoundingBox3D bbox( getViewerState().mVolumeRenderConfig.primaryImageData->GetMinimum(), 
						getViewerState().mVolumeRenderConfig.primaryImageData->GetMaximum() );
			GL_CHECKED_CALL( glEnable( GL_DEPTH_TEST ) );
			GL_CHECKED_CALL( glDisable( GL_LIGHTING ) );
			getViewerState().mBasicCgEffect.SetParameter( "gViewSetup", getViewerState().glViewSetup );
			if ( getViewerState().mEnableVolumeBoundingBox ) {
				glColor3f( 1.0f, 0.0f, 0.0f );
				getViewerState().mBasicCgEffect.ExecuteTechniquePass( "Basic", boost::bind( &M4D::GLDrawBoundingBox, bbox ) );
			}
			//Draw cut plane if enabled TODO - set color
			getViewerState().mBasicCgEffect.ExecuteTechniquePass( 
				"Basic", boost::bind( &M4D::GUI::Viewer::handleCutPlane, getViewerState().mVolumeRenderConfig.enableCutPlane, bbox, getViewerState().mVolumeRenderConfig.cutPlane, M4D::RGBAf( 0.0f, 1.0f, 0.0f, 1.0f ) ) 
				);
		
			/*GL_CHECKED_CALL( glEnable( GL_LIGHTING ) );
			GL_CHECKED_CALL( glEnable( GL_LIGHT0 ) );
			GL_CHECKED_CALL( glLightfv( GL_LIGHT0, GL_AMBIENT, Vector4f( 0.25f, 0.25f, 0.25f, 1.0f ).GetData() ) );
			GL_CHECKED_CALL( glLightfv( GL_LIGHT0, GL_DIFFUSE, Vector4f( 1.0f, 1.0f, 1.0f, 1.0f ).GetData() ) );
			GL_CHECKED_CALL( glLightfv( GL_LIGHT0, GL_POSITION, Vector4f( getViewerState().mVolumeRenderConfig.lightPosition, 1.0f ).GetData() ) );*/

			//LOG( getViewerState().glViewSetup );
			if ( mRenderingExtension && (vt3D | mRenderingExtension->getAvailableViewTypes()) ) {
				getViewerState().mBasicCgEffect.ExecuteTechniquePass( "Basic", boost::bind( &M4D::GUI::Viewer::RenderingExtension::preRender3D, mRenderingExtension ) );
				//mRenderingExtension->preRender3D();	
			}

			try {
				getViewerState().mVolumeRenderer.Render( getViewerState().mVolumeRenderConfig, getViewerState().glViewSetup );
			}catch( std::exception &e ) {
				LOG( e.what() );
			}

			GL_CHECKED_CALL( glClear( GL_DEPTH_BUFFER_BIT ) );//TODO disable depth storing during volume rendering
			if ( mRenderingExtension && (vt3D | mRenderingExtension->getAvailableViewTypes()) ) {
				getViewerState().mBasicCgEffect.ExecuteTechniquePass( "Basic", boost::bind( &M4D::GUI::Viewer::RenderingExtension::postRender3D, mRenderingExtension ) );
				//mRenderingExtension->postRender3D();
			}
		}
		break;
	case vt2DAlignedSlices:
		{
			ASSERT( getViewerState().m2DMultiSliceGrid[0] > 0 && getViewerState().m2DMultiSliceGrid[1] > 0 );
			int subVPortW = width() / getViewerState().m2DMultiSliceGrid[1];
			int subVPortH = height() / getViewerState().m2DMultiSliceGrid[0];
			//size_t sliceOffset = 0;
			Renderer::SliceRenderer::RenderingConfiguration config = getViewerState().mSliceRenderConfig;
			for ( unsigned j = 0; j < getViewerState().m2DMultiSliceGrid[0]; ++j ) {
				for ( unsigned i = 0; i < getViewerState().m2DMultiSliceGrid[1]; ++i ) {
					
					GL_CHECKED_CALL( glViewport( i * subVPortW, j * subVPortH, subVPortW, subVPortH ) );
					//SetToViewConfiguration2D( config.viewConfig );
					
					try {
						getViewerState().mSliceRenderer.Render( config, getViewerState().glViewSetup );
					}catch( std::exception &e ) {
						LOG( e.what() );
					}
					
					glClear( GL_DEPTH_BUFFER_BIT );
					/*if ( mRenderingExtension && (vt2DAlignedSlices | mRenderingExtension->getAvailableViewTypes()) ) {
						
					}*/
					
					if ( mRenderingExtension && (vt2DAlignedSlices | mRenderingExtension->getAvailableViewTypes()) ) {
						CartesianPlanes plane = config.plane;
						Vector3f realSlices = config.getCurrentRealSlice();
						Vector3f hextents = 0.5f * getViewerState().getMinimalElementExtents();
						getViewerState().mSceneSlicingCgEffect.SetParameter( "gPlaneNormal", getViewerState().mSliceRenderConfig.sliceNormal );
						getViewerState().mSceneSlicingCgEffect.SetParameter( "gPlanePoint", getViewerState().mSliceRenderConfig.sliceCenter );
						getViewerState().mSceneSlicingCgEffect.SetParameter( "gPlaneWidth", 2*hextents[plane] );
						getViewerState().mSceneSlicingCgEffect.SetParameter( "gViewSetup", getViewerState().glViewSetup );
						getViewerState().mSceneSlicingCgEffect.ExecuteTechniquePass( 
									"SceneSlicing", 
									boost::bind( &M4D::GUI::Viewer::RenderingExtension::render2DAlignedSlices,
									mRenderingExtension,
									config.currentSlice[ config.plane ], 
									Vector2f( realSlices[plane] - hextents[plane], realSlices[plane] + hextents[plane] ), 
									config.plane
									) );
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
			Vector3f direction;
			//LOG( mViewerState->glViewSetup );
			try{
				direction = getDirectionFromScreenCoordinatesAndCameraPosition(
				       	Vector2f( event->posF().x(), mViewerState->glViewSetup.viewport[3] - event->posF().y() ), 
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
			Vector3f eye = getViewerState().mSliceRenderConfig.camera.GetEyePosition();
			Vector3f target = getViewerState().mSliceRenderConfig.camera.GetTargetPosition();
			Vector3f dir = target - eye;
			VectorNormalization( dir );
			
			int x = event->pos().x() / subVPortW;
			int y = event->pos().y() / subVPortH;
			Vector3f slicePoint = getViewerState().mSliceRenderConfig.sliceCenter;
			Vector3d pom = getPointFromScreenCoordinates( Vector2f( event->posF().x() - subVPortW * x, mViewerState->glViewSetup.viewport[3] - event->posF().y() + subVPortH * y ), mViewerState->glViewSetup );
			//LOG( event->posF().x() << ";  " << event->posF().y() );
			Vector3f intersection;
			IntersectionResult res = AxisPlaneIntersection( 
					Vector3f( pom ), 
					dir,
					slicePoint, 
					dir,
					intersection
					);
			//LOG( "Intersection res = " << res << "; " << intersection );
			
			/*Vector2f pos = GetRealCoordinatesFromScreen( 
				Vector2f( event->posF().x(), event->posF().y() ), 
				getViewerState().mWindowSize, 
				getViewerState().mSliceRenderConfig.viewConfig 
				);
			float32 realSlice = getCurrentRealSlice();*/
			//Vector3f position = VectorInsertDimension( pos, realSlice, getCurrentViewPlane() );
			Vector3f position = intersection;
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
	try {
		if( TryGetAndLockAllAvailableInputs() == 0 ){
			return false;
		};
	} catch (...) {
		return false;
	}

	if ( ! mInputDatasets[0] ) return false;
	M4D::Imaging::AImageDim<3>::ConstPtr primaryImage = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] );
	
	if ( ! primaryImage ) return false;
	
	getViewerState().mPrimaryImageExtents = primaryImage->GetImageExtentsRecord();
	getViewerState().mPrimaryImageTexture = OpenGLManager::getInstance()->getTextureFromImage( *primaryImage );
	

	ReleaseAllInputs();

	getViewerState().mSliceRenderConfig.currentSlice = getViewerState().mPrimaryImageExtents.minimum;
	getViewerState().mSliceRenderConfig.primaryImageData = &(getViewerState().mPrimaryImageTexture->GetDimensionedInterface<3>());
	getViewerState().mVolumeRenderConfig.primaryImageData = &(getViewerState().mPrimaryImageTexture->GetDimensionedInterface<3>());

	getViewerState().mVolumeRenderConfig.camera.SetTargetPosition( 0.5f * (getViewerState().mPrimaryImageTexture->GetDimensionedInterface< 3 >().GetMaximum() + getViewerState().mPrimaryImageTexture->GetDimensionedInterface< 3 >().GetMinimum()) );
	getViewerState().mVolumeRenderConfig.camera.SetFieldOfView( 45.0f );
	getViewerState().mVolumeRenderConfig.camera.SetEyePosition( Vector3f( 0.0f, 0.0f, 750.0f ) );
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


} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/


#ifdef USE_CG

#include "GUI/utils/OGLDrawing.h"
#include "GUI/utils/QtM4DTools.h"
#include "common/MathTools.h"
#include "GUI/widgets/GeneralViewer.h"
#include "Imaging/ImageFactory.h"
#include "GUI/utils/CameraManipulator.h"
#include "GUI/utils/ViewerManager.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{

ViewerController::ViewerController()
{
	mCameraOrbitButton = Qt::MidButton;
	mLUTSetMouseButton = Qt::RightButton;
	mFastSliceChangeMouseButton = Qt::MidButton;

	mTimer.setSingleShot( false );
	QObject::connect( &mTimer, SIGNAL(timeout()), this, SLOT( timerCall() ) );
}

bool
ViewerController::mouseMoveEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event )
{
	ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );

	QPoint diff = mTrackInfo.trackUpdate( event->pos(), event->globalPos() );
	if ( state.viewType == vt3D && mInteractionMode == imORBIT_CAMERA ) {
		state.getViewerWindow< GeneralViewer >().cameraOrbit( Vector2f( diff.x() * -0.02f, diff.y() * -0.02f ) );
		return true;
	}
	if ( mInteractionMode == imLUT_SETTING ) {
		Vector2f oldVal = state.getViewerWindow< GeneralViewer >().getLUTWindow();
		state.getViewerWindow< GeneralViewer >().setLUTWindow( oldVal + Vector2f( diff.x(), diff.y() ) );
		return true;
	}
	if ( mInteractionMode == imFAST_SLICE_CHANGE ) {
		int speed = mTrackInfo.mStartLocalPosition.y() - event->pos().y();
		mTmpViewer = &(state.getViewerWindow< GeneralViewer >());
		mPositive = speed > 0;
		if( speed != 0 ) {
			float ms = 1000.0f / Abs(speed);
			mTimer.setInterval( Max<int>( static_cast<int>( ms ), 10 ) );
			mTimer.start();
			timerCall();
		} else {
			mTimer.stop();
		}
		return true;
	}
	return false;
}

bool	
ViewerController::mouseDoubleClickEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event )
{
	ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );
	if ( state.viewType == vt2DAlignedSlices ) {
		state.getViewerWindow< GeneralViewer >().switchToNextPlane();

		event->accept();
		return true;
	}

	return false;
}

bool
ViewerController::mousePressEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event )
{
	ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );

	mTrackInfo.startTracking( event->pos(), event->globalPos() );
	if ( state.viewType == vt3D ) {
		if( event->button() == mCameraOrbitButton ) {
			mInteractionMode = imORBIT_CAMERA;
			return true;
		}
	}
	if ( state.viewType == vt2DAlignedSlices ) {
		if( event->button() == mFastSliceChangeMouseButton ) {
			mInteractionMode = imFAST_SLICE_CHANGE;
			return true;
		}
	}
	if ( state.colorTransform == M4D::GUI::Renderer::ctLUTWindow || state.colorTransform == M4D::GUI::Renderer::ctMaxIntensityProjection ) {
		if( event->button() == mLUTSetMouseButton ) {
			mInteractionMode = imLUT_SETTING;
			return true;
		}
	}

	return false;
}

bool
ViewerController::mouseReleaseEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event )
{
	//ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );
	if ( (mInteractionMode == imORBIT_CAMERA && event->button() == mCameraOrbitButton)
	  || (mInteractionMode == imLUT_SETTING && event->button() == mLUTSetMouseButton) ) 
	{
		mInteractionMode = imNONE;
		return true;
	}
	if ( mInteractionMode == imFAST_SLICE_CHANGE && event->button() == mFastSliceChangeMouseButton ) {
		mTimer.stop();
		mInteractionMode = imNONE;
		return true;
	}
	return false;
}

bool
ViewerController::wheelEvent ( BaseViewerState::Ptr aViewerState, QWheelEvent * event )
{
	ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );

	//int numDegrees = event->delta() / 8;
	//int numSteps = numDegrees / 15;
	
	if ( state.viewType == vt3D ) {
		float dollyRatio = 1.1f;
		if ( event->delta() > 0 ) {
			dollyRatio = 1.0f/dollyRatio;
		}
		state.getViewerWindow< GeneralViewer >().cameraDolly( dollyRatio );
		event->accept();
		return true;
	}
	if ( state.viewType == vt2DAlignedSlices ) {
		state.getViewerWindow< GeneralViewer >().changeCurrentSlice( event->delta() > 0 ? 1: -1 );

		event->accept();
		return true;
	}

	return false;
}


void
ViewerController::timerCall()
{
	if ( mInteractionMode == imFAST_SLICE_CHANGE ) {
		ASSERT(mTmpViewer);
		mTmpViewer->changeCurrentSlice( mPositive ? 1: -1 );
	}
}


//********************************************************************************************

GeneralViewer::GeneralViewer( QWidget *parent ): PredecessorType( parent ), _prepared( false )
{
	ViewerState * state = new ViewerState;

	state->mSliceRenderConfig.colorTransform = M4D::GUI::Renderer::ctLUTWindow;
	state->mSliceRenderConfig.plane = XY_PLANE;

	state->mVolumeRenderConfig.colorTransform = M4D::GUI::Renderer::ctMaxIntensityProjection;
	state->mVolumeRenderConfig.sampleCount = 200;
	state->mVolumeRenderConfig.shadingEnabled = true;
	state->mVolumeRenderConfig.jitterEnabled = true;

	state->viewerWindow = this;

	//state->backgroundColor = QColor( 20, 10, 90);
	state->backgroundColor = QColor( 0, 0, 0);

	state->availableViewTypes = 5;
	state->viewType = vt2DAlignedSlices;

	mViewerState = BaseViewerState::Ptr( state );



	setColorTransformType( M4D::GUI::Renderer::ctLUTWindow );

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
	getViewerState().mTFunctionBuffer = aTFunctionBuffer;
	
	makeCurrent();
	getViewerState().mTransferFunctionTexture = CreateGLTransferFunctionBuffer1D( *aTFunctionBuffer );
	doneCurrent();

	getViewerState().mSliceRenderConfig.transferFunction = getViewerState().mTransferFunctionTexture.get();
	getViewerState().mVolumeRenderConfig.transferFunction = getViewerState().mTransferFunctionTexture.get();

	notifyAboutSettingsChange();
	update();
}

void
GeneralViewer::setCurrentSlice( int32 slice )
{
	CartesianPlanes plane = getViewerState().mSliceRenderConfig.plane;
	getViewerState().mSliceRenderConfig.currentSlice[ plane ] = Max( 
								Min( getViewerState()._regionMax[plane]-1, slice ), 
								getViewerState()._regionMin[plane] );
	notifyAboutSettingsChange();
	update();
}

void
GeneralViewer::switchToNextPlane()
{
	getViewerState().mSliceRenderConfig.plane = NextCartesianPlane( getViewerState().mSliceRenderConfig.plane );
	notifyAboutSettingsChange();
	update();
}

void
GeneralViewer::setCurrentViewPlane( CartesianPlanes aPlane )
{
	getViewerState().mSliceRenderConfig.plane = aPlane;
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

void
GeneralViewer::changeCurrentSlice( int32 diff )
{
	setCurrentSlice( diff + getCurrentSlice() );
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
			return QString::fromStdString( (*idList)[ i ].name );
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

void
GeneralViewer::cameraOrbit( Vector2f aAngles )
{
	getViewerState().mVolumeRenderConfig.camera.YawAround( aAngles[0] );
	getViewerState().mVolumeRenderConfig.camera.PitchAround( aAngles[1] );
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
		strList << QString::fromStdString( (*idList)[ i ].name );
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
	std::string name = aColorTransformName.toStdString();
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
		if( name == (*idList)[ i ].name ) {
			setColorTransformType( (*idList)[ i ].id );
			D_PRINT( "Setting color transform : " << aColorTransformName.toStdString() );
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
GeneralViewer::resetView()
{
	Vector3f pos = getViewerState().mVolumeRenderConfig.camera.GetTargetPosition();
	pos[1] += -550;
	getViewerState().mVolumeRenderConfig.camera.SetEyePosition( pos, Vector3f( 0.0f, 0.0f, 1.0f ) );
	
	update();
}

//********************************************************************************

void
GeneralViewer::initializeRenderingEnvironment()
{
	getViewerState().mSliceRenderer.Initialize();
	getViewerState().mVolumeRenderer.Initialize();
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
			SetViewAccordingToCamera( getViewerState().mVolumeRenderConfig.camera );
		}
		break;
	case vt2DAlignedSlices:
		{
			zoomFit();
			SetToViewConfiguration2D( getViewerState().mSliceRenderConfig.viewConfig );
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
			glEnable( GL_DEPTH_TEST );
			if ( getViewerState().mEnableVolumeBoundingBox ) {
				glColor3f( 1.0f, 0.0f, 0.0f );
				M4D::GLDrawBoundingBox( getViewerState().mVolumeRenderConfig.imageData->GetMinimum(), getViewerState().mVolumeRenderConfig.imageData->GetMaximum() );
			}
			if ( mRenderingExtension && (vt3D | mRenderingExtension->getAvailableViewTypes()) ) {
				mRenderingExtension->preRender3D();	
			}

			getViewerState().mVolumeRenderer.Render( getViewerState().mVolumeRenderConfig, false );

			glClear( GL_DEPTH_BUFFER_BIT );
			if ( mRenderingExtension && (vt3D | mRenderingExtension->getAvailableViewTypes()) ) {
				mRenderingExtension->postRender3D();	
			}
		}
		break;
	case vt2DAlignedSlices:
		{
			getViewerState().mSliceRenderer.Render( getViewerState().mSliceRenderConfig, false );

			glClear( GL_DEPTH_BUFFER_BIT );
			if ( mRenderingExtension && (vt2DAlignedSlices | mRenderingExtension->getAvailableViewTypes()) ) {
				CartesianPlanes plane = getViewerState().mSliceRenderConfig.plane;
				Vector3f realSlices = getViewerState().mSliceRenderConfig.getCurrentRealSlice();
				Vector3f hextents = 0.5f * getViewerState()._elementExtents;

				mRenderingExtension->render2DAlignedSlices( getViewerState().mSliceRenderConfig.currentSlice[ getViewerState().mSliceRenderConfig.plane ], 
						Vector2f( realSlices[plane] - hextents[plane], realSlices[plane] + hextents[plane] ), 
						getViewerState().mSliceRenderConfig.plane 
						);	
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
		TryGetAndLockAllInputs();
	} catch (...) {
		return false;
	}

	getViewerState()._regionMin = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetMinimum();
	getViewerState()._regionMax = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetMaximum();
	getViewerState()._regionRealMin = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetRealMinimum();
	getViewerState()._regionRealMax = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetRealMaximum();
	getViewerState()._elementExtents = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetElementExtents();

	getViewerState().mSliceRenderConfig.currentSlice = getViewerState()._regionMin;

	//getViewerState()._textureData = CreateTextureFromImage( *(M4D::Imaging::AImage::Cast( mInputDatasets[0] )->GetAImageRegion()), true ) ;
	getViewerState()._textureData = OpenGLManager::getInstance()->getTextureFromImage( *(M4D::Imaging::AImage::Cast( mInputDatasets[0] )) );
	ReleaseAllInputs();


	getViewerState().mSliceRenderConfig.imageData = &(getViewerState()._textureData->GetDimensionedInterface<3>());
	getViewerState().mVolumeRenderConfig.imageData = &(getViewerState()._textureData->GetDimensionedInterface<3>());

	getViewerState().mVolumeRenderConfig.camera.SetTargetPosition( 0.5f * (getViewerState()._textureData->GetDimensionedInterface< 3 >().GetMaximum() + getViewerState()._textureData->GetDimensionedInterface< 3 >().GetMinimum()) );
	getViewerState().mVolumeRenderConfig.camera.SetFieldOfView( 45.0f );
	getViewerState().mVolumeRenderConfig.camera.SetEyePosition( Vector3f( 0.0f, 0.0f, 750.0f ) );
	resetView();

	_prepared = true;
	return true;
}

void	
GeneralViewer::zoomFit( ZoomType zoomType )
{
	getViewerState().mSliceRenderConfig.viewConfig = GetOptimalViewConfiguration(
			VectorPurgeDimension( getViewerState()._regionRealMin, getViewerState().mSliceRenderConfig.plane ), 
			VectorPurgeDimension( getViewerState()._regionRealMax, getViewerState().mSliceRenderConfig.plane ),
			Vector< uint32, 2 >( (uint32)width(), (uint32)height() ), 
			zoomType );
	emit settingsChanged();
	update();
}


} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/


#endif /*USE_CG*/

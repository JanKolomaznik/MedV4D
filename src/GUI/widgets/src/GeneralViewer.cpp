#ifdef USE_CG

#include "GUI/utils/OGLDrawing.h"
#include "GUI/utils/QtM4DTools.h"
#include "common/MathTools.h"
#include "GUI/widgets/GeneralViewer.h"
#include "Imaging/ImageFactory.h"
#include "GUI/utils/CameraManipulator.h"

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

	state->availableViewTypes = 7;
	state->viewType = vt2DAlignedSlices;

	mViewerState = BaseViewerState::Ptr( state );



	setColorTransformType( M4D::GUI::Renderer::ctLUTWindow );

}


void
GeneralViewer::setLUTWindow( Vector2f window )
{
	getViewerState().mSliceRenderConfig.lutWindow = window;
	getViewerState().mVolumeRenderConfig.lutWindow = window;
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

	update();
}

void
GeneralViewer::setCurrentSlice( int32 slice )
{
	CartesianPlanes plane = getViewerState().mSliceRenderConfig.plane;
	getViewerState().mSliceRenderConfig.currentSlice[ plane ] = Max( 
								Min( getViewerState()._regionMax[plane]-1, slice ), 
								getViewerState()._regionMin[plane] );
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
	if( !IsDataPrepared() && !PrepareData() ) {
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
			if ( getViewerState().mEnableVolumeBoundingBox ) {
				glColor3f( 1.0f, 0.0f, 0.0f );
				M4D::GLDrawBoundingBox( getViewerState().mVolumeRenderConfig.imageData->GetMinimum(), getViewerState().mVolumeRenderConfig.imageData->GetMaximum() );
			}

			getViewerState().mVolumeRenderer.Render( getViewerState().mVolumeRenderConfig, false );
		}
		break;
	case vt2DAlignedSlices:
		{
			getViewerState().mSliceRenderer.Render( getViewerState().mSliceRenderConfig, false );
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

	getViewerState()._textureData = CreateTextureFromImage( *(M4D::Imaging::AImage::Cast( mInputDatasets[0] )->GetAImageRegion()), true ) ;

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
}


} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/


#endif /*USE_CG*/

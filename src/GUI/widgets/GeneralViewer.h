#ifdef USE_CG
#ifndef GENERAL_VIEWER_H
#define GENERAL_VIEWER_H

#include "Imaging/Imaging.h"

#include "GUI/utils/ViewConfiguration.h"
#include "GUI/utils/IUserEvents.h"
#include "GUI/utils/TransferFunctionBuffer.h"
#include "GUI/utils/CgShaderTools.h"
#include "GUI/utils/GLTextureImage.h"
#include "GUI/utils/FrameBufferObject.h"
#include "GUI/widgets/AGUIViewer.h"
#include "GUI/widgets/ViewerConstructionKit.h"

#include "GUI/widgets/AGLViewer.h"
#include "GUI/renderers/SliceRenderer.h"
#include "GUI/renderers/VolumeRenderer.h"

#include <QtGui>
#include <QtOpenGL>
#include <boost/shared_ptr.hpp>
#include <boost/cast.hpp>
#include <map>


namespace M4D
{
namespace GUI
{
namespace Viewer
{

class GeneralViewer;

class ViewerState : public BaseViewerState
{
public:
	typedef boost::shared_ptr< ViewerState > Ptr;
	
	GLTextureImage::Ptr	_textureData;

	unsigned colorTransform;

	Vector< float, 3 > 			_regionRealMin;
	Vector< float, 3 >			_regionRealMax;
	Vector< float, 3 >			_elementExtents;
	Vector< int32, 3 > 			_regionMin;
	Vector< int32, 3 >			_regionMax;

	TransferFunctionBuffer1D::Ptr 		mTFunctionBuffer;
	GLTransferFunctionBuffer1D::Ptr 	mTransferFunctionTexture;

	M4D::GUI::Renderer::SliceRenderer	mSliceRenderer;
	M4D::GUI::Renderer::SliceRenderer::RenderingConfiguration mSliceRenderConfig;

	M4D::GUI::Renderer::VolumeRenderer	mVolumeRenderer;
	M4D::GUI::Renderer::VolumeRenderer::RenderingConfiguration mVolumeRenderConfig;
	bool 					mEnableVolumeBoundingBox;
	
};

struct MouseTrackInfo
{
	void
	startTracking( QPoint aLocalPosition, QPoint aGlobalPosition ) 
	{
		mStartLocalPosition = mLastLocalPosition = aLocalPosition;

		mStartGlobalPosition = mLastGlobalPosition = aGlobalPosition;
	}

	QPoint
	trackUpdate( QPoint aLocalPosition, QPoint aGlobalPosition )
	{
		QPoint diff = aLocalPosition - mLastLocalPosition;
		mLastLocalPosition = aLocalPosition;

		mLastGlobalPosition = aGlobalPosition;
		return diff;
	}

	QPoint	mStartLocalPosition;
	QPoint	mLastLocalPosition;

	QPoint	mStartGlobalPosition;
	QPoint	mLastGlobalPosition;
};

class ViewerController
{
public:
	typedef boost::shared_ptr< ViewerController > Ptr;
	enum InteractionMode { 
		imNONE,
		imORBIT_CAMERA,
		imLUT_SETTING
	};

	ViewerController()
	{
		mCameraOrbitButton = Qt::MidButton;
		mLUTSetMouseButton = Qt::RightButton;
	}

	bool
	mouseMoveEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event )
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
	mouseDoubleClickEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event )
	{
		ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );

		return false;
	}

	bool
	mousePressEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event )
	{
		ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );

		mTrackInfo.startTracking( event->pos(), event->globalPos() );
		if ( state.viewType == vt3D ) {
			if( event->button() == mCameraOrbitButton ) {
				mInteractionMode = imORBIT_CAMERA;
				return true;
			}
		}

		if ( state.colorTransform == ctLUTWindow || state.colorTransform == ctMaxIntensityProjection ) {
			if( event->button() == mLUTSetMouseButton ) {
				mInteractionMode = imLUT_SETTING;
				return true;
			}
		}

		return false;
	}

	bool
	mouseReleaseEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event )
	{
		ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );
		if ( (mInteractionMode == imORBIT_CAMERA && event->button() == mCameraOrbitButton)
		  || (mInteractionMode == imLUT_SETTING && event->button() == mLUTSetMouseButton) ) 
		{
			mInteractionMode = imNONE;
			return true
		}
		return false;
	}

	bool
	wheelEvent ( BaseViewerState::Ptr aViewerState, QWheelEvent * event )
	{
		ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );

		int numDegrees = event->delta() / 8;
		int numSteps = numDegrees / 15;
		
		if ( state.viewType == vt3D ) {
			float dollyRatio = 1.1f;
			if ( event->delta() > 0 ) {
				dollyRatio = 1.0f/dollyRatio;
			}
			state.getViewerWindow< GeneralViewer >().cameraDolly( dollyRatio );
			event->accept();
			return true;
		}

		return false;
	}

protected:
	Qt::MouseButton	mCameraOrbitButton;

	InteractionMode mInteractionMode;
	MouseTrackInfo	mTrackInfo;
};




class GeneralViewer : 
	public ViewerConstructionKit<   AGLViewer, 
					PortInterfaceHelper< boost::mpl::vector< M4D::Imaging::AImage > >
					>
{
	Q_OBJECT;
public:
	typedef ViewerConstructionKit<  AGLViewer, 
					PortInterfaceHelper< boost::mpl::vector< M4D::Imaging::AImage > >
					>	PredecessorType;


	GeneralViewer( QWidget *parent = NULL );


	void
	setLUTWindow( float32 center, float32 width );

	void
	setLUTWindow( Vector2f window );

	Vector2f
	getLUTWindow() const;

	void
	setTransferFunctionBuffer( TransferFunctionBuffer1D::Ptr aTFunctionBuffer );

	void
	setCurrentSlice( int32 slice );

	bool
	isColorTransformAvailable( unsigned aTransformType );

	/*int
	GetRendererType()
	{
		return mRendererType;//_renderer.GetRendererType();
	}*/

	int
	getColorTransformType()
	{
		return getViewerState().colorTransform;//mSliceRenderConfig.colorTransform;//_renderer.GetColorTransformType();
	}


	void
	ReceiveMessage( 
		M4D::Imaging::PipelineMessage::Ptr 			msg, 
		M4D::Imaging::PipelineMessage::MessageSendStyle 	sendStyle, 
		M4D::Imaging::FlowDirection				direction
	)
	{
		PrepareData();
	}

	bool
	isShadingEnabled() const
	{
		return getViewerState().mVolumeRenderConfig.shadingEnabled;
	}

	bool
	isJitteringEnabled() const
	{
		return getViewerState().mVolumeRenderConfig.jitterEnabled;
	}

	void
	cameraOrbit( Vector2f aAngles )
	{
		getViewerState().mVolumeRenderConfig.camera.YawAround( aAngles[0] );
		getViewerState().mVolumeRenderConfig.camera.PitchAround( aAngles[1] );
		update();
	}

	void
	cameraDolly( float aDollyRatio )
	{
		DollyCamera( getViewerState().mVolumeRenderConfig.camera, aDollyRatio );
		update();
	}
public slots:
	void
	setViewType( int aViewType )
	{
		getViewerState().mViewType = aViewType;
		//TODO 

		update();

		emit ViewTypeChanged( aViewType );
	}

	void
	setColorTransformType( int aColorTransform )
	{
		//TODO 
		getViewerState().colorTransform = aColorTransform;
		getViewerState().mSliceRenderConfig.colorTransform = aColorTransform;
		getViewerState().mVolumeRenderConfig.colorTransform = aColorTransform;

		update();

		emit ColorTransformTypeChanged( aColorTransform );
	}

	void
	fineRender()
	{
		//_renderer.FineRender();
		update();
	}

	void
	enableShading( bool aEnable )
	{
		getViewerState().mVolumeRenderConfig.shadingEnabled = aEnable;
		update();
	}

	void
	enableJittering( bool aEnable )
	{
		getViewerState().mVolumeRenderConfig.jitterEnabled = aEnable;
		update();
	}

	
	void
	resetView()
	{
		Vector3f pos = getViewerState().mVolumeRenderConfig.camera.GetTargetPosition();
		pos[1] += -550;
		getViewerState().mVolumeRenderConfig.camera.SetEyePosition( pos, Vector3f( 0.0f, 0.0f, 1.0f ) );
		
		update();
	}

signals:
	void
	SettingsChanged();

	void
	ViewTypeChanged( int aRendererType );

	void
	ColorTransformTypeChanged( int aColorTransform );

	void
	MouseInfoUpdate( const QString &aInfo );


protected:

	void
	initializeRenderingEnvironment();

	bool
	preparedForRendering();

	void
	prepareForRenderingStep();

	void
	render();

	void
	finalizeAfterRenderingStep();

//******** TMP ************
	bool
	IsDataPrepared();

	bool
	PrepareData();

	

	bool _prepared;
//******** TMP ************

private:
	ViewerState &
	getViewerState()
	{
		ASSERT( mViewerState );
		return *(boost::polymorphic_downcast< ViewerState *>( mViewerState.get() ) ); 
	}
};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/



#endif /*GENERAL_VIEWER_H*/



#endif /*USE_CG*/

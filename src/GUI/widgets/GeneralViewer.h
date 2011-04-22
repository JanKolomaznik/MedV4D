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

class ViewerState : public BaseViewerState
{
public:
	typedef boost::shared_ptr< ViewerState > Ptr;
	
	GLTextureImage::Ptr	_textureData;


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

class ViewerController
{
public:
	typedef boost::shared_ptr< ViewerController > Ptr;
	enum InteractionMode { 
		imNONE,
		imORBIT_CAMERA 
	};

	bool
	mouseMoveEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event )
	{
		ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );
		if ( state.viewType == vt3D ) {
			QPoint tmp = event->globalPos();
			switch ( mInteractionMode ) {
			case imORBIT_CAMERA: {
					QPoint diff = tmp - mLastPoint;   
					state.mVolumeRenderConfig.camera.YawAround( diff.x() * -0.02f );
					state.mVolumeRenderConfig.camera.PitchAround( diff.y() * -0.02f );
					break;
				}
			default:
				break;
			}
			mLastPoint = event->globalPos();
			state.viewerWindow->update();
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
		if ( state.viewType == vt3D ) {
			mClickPosition = event->globalPos();
			mLastPoint = mClickPosition;
			if( event->button() == Qt::LeftButton ) {
				mInteractionMode = imORBIT_CAMERA;
				return true;
			}
		}
		return false;
	}

	bool
	mouseReleaseEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event )
	{
		ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );
		if ( state.viewType == vt3D ) {
			mInteractionMode = imNONE;
		}
		return false;
	}

	bool
	wheelEvent ( BaseViewerState::Ptr aViewerState, QWheelEvent * event )
	{
		ViewerState &state = *(boost::polymorphic_downcast< ViewerState *>( aViewerState.get() ) );

		return false;
	}

protected:
	InteractionMode mInteractionMode;
	QPoint					mClickPosition;
	QPoint					mLastPoint;
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
	SetLUTWindow( float32 center, float32 width );

	void
	SetLUTWindow( Vector2f window );

	void
	SetTransferFunctionBuffer( TransferFunctionBuffer1D::Ptr aTFunctionBuffer );

	void
	SetCurrentSlice( int32 slice );

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

	void
	ResetView()
	{
		Vector3f pos = getViewerState().mVolumeRenderConfig.camera.GetTargetPosition();
		pos[1] += -550;
		getViewerState().mVolumeRenderConfig.camera.SetEyePosition( pos, Vector3f( 0.0f, 0.0f, 1.0f ) );
		
		update();
	}

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

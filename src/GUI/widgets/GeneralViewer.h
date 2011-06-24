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
#include "GUI/renderers/RendererTools.h"
#include "GUI/renderers/SliceRenderer.h"
#include "GUI/renderers/VolumeRenderer.h"

#include <QtGui>
#include <QtOpenGL>
#include <boost/shared_ptr.hpp>
#include <boost/cast.hpp>
#include <map>

#include "common/Types.h"


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

class ViewerController: public AViewerController
{
	Q_OBJECT;
public:
	typedef boost::shared_ptr< ViewerController > Ptr;
	enum InteractionMode { 
		imNONE,
		imORBIT_CAMERA,
		imLUT_SETTING,
		imFAST_SLICE_CHANGE
	};

	ViewerController();

	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QMouseEvent * event );

	bool	
	mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QMouseEvent * event );

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QMouseEvent * event );

	bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QMouseEvent * event );

	bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event );

protected slots:
	virtual void
	timerCall();

protected:
	Qt::MouseButton	mCameraOrbitButton;
	Qt::MouseButton	mLUTSetMouseButton;
	Qt::MouseButton	mFastSliceChangeMouseButton;

	InteractionMode mInteractionMode;
	MouseTrackInfo	mTrackInfo;

	QTimer	mTimer;
	GeneralViewer *mTmpViewer;
	bool mPositive;
};

class RenderingExtension
{
public:
	typedef boost::shared_ptr< RenderingExtension > Ptr;

	virtual unsigned
	getAvailableViewTypes()const = 0;

	virtual void
	render2DAlignedSlices( int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane ) = 0;

	virtual void
	preRender3D() = 0;

	virtual void
	postRender3D() = 0;
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

	void
	switchToNextPlane();

	CartesianPlanes
	getCurrentViewPlane()const;

	void
	setCurrentViewPlane( CartesianPlanes aPlane );

	int32
	getCurrentSlice()const;

	void
	changeCurrentSlice( int32 diff );
	
	bool
	isColorTransformAvailable( unsigned aTransformType );

	/*int
	GetRendererType()
	{
		return mRendererType;//_renderer.GetRendererType();
	}*/

	int
	getColorTransformType();

	QString
	getColorTransformName();

	void
	ReceiveMessage( 
		M4D::Imaging::PipelineMessage::Ptr 			msg, 
		M4D::Imaging::PipelineMessage::MessageSendStyle 	sendStyle, 
		M4D::Imaging::FlowDirection				direction
		);

	bool
	isShadingEnabled() const;

	bool
	isJitteringEnabled() const;

	void
	cameraOrbit( Vector2f aAngles );

	void
	cameraOrbitAbsolute( Vector2f aAngles );

	void
	cameraDolly( float aDollyRatio );

	ViewType
	getViewType()const;

	QStringList 
	getAvailableColorTransformationNames();

	GUI::Renderer::ColorTransformNameIDList
	getAvailableColorTransformations();
	
	void
	setRenderingExtension( RenderingExtension::Ptr aRenderingExtension );
public slots:
	void
	setViewType( int aViewType );

	void
	setColorTransformType( const QString & aColorTransformName );

	void
	setColorTransformType( int aColorTransform );

	void
	fineRender();

	void
	enableShading( bool aEnable );

	void
	enableJittering( bool aEnable );

	
	void
	resetView();

	void
	zoomFit( ZoomType zoomType = ztFIT );

signals:
	void
	settingsChanged();

	void
	ViewTypeChanged( int aRendererType );

	void
	ColorTransformTypeChanged( int aColorTransform );

	void
	MouseInfoUpdate( const QString &aInfo );


protected:
	void
	notifyAboutSettingsChange();

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

	RenderingExtension::Ptr mRenderingExtension;
private:
	ViewerState &
	getViewerState()
	{
		ASSERT( mViewerState );
		return *(boost::polymorphic_downcast< ViewerState *>( mViewerState.get() ) ); 
	}
	const ViewerState &
	getViewerState() const
	{
		ASSERT( mViewerState );
		return *(boost::polymorphic_downcast< const ViewerState *>( mViewerState.get() ) ); 
	}
};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/



#endif /*GENERAL_VIEWER_H*/



#endif /*USE_CG*/

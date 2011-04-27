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
public:
	typedef boost::shared_ptr< ViewerController > Ptr;
	enum InteractionMode { 
		imNONE,
		imORBIT_CAMERA,
		imLUT_SETTING
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

protected:
	Qt::MouseButton	mCameraOrbitButton;
	Qt::MouseButton	mLUTSetMouseButton;

	InteractionMode mInteractionMode;
	MouseTrackInfo	mTrackInfo;
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
	switchToNextPlane()
	{
		getViewerState().mSliceRenderConfig.plane = NextCartesianPlane( getViewerState().mSliceRenderConfig.plane );
		update();
	}

	int32
	getCurrentSlice()const
	{
		CartesianPlanes plane = getViewerState().mSliceRenderConfig.plane;
		return getViewerState().mSliceRenderConfig.currentSlice[ plane ];
	}

	void
	changeCurrentSlice( int32 diff )
	{
		setCurrentSlice( diff + getCurrentSlice() );
	}

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

	QString
	getColorTransformName()
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
				return QString::fromStdWString( (*idList)[ i ].name );
			}
		}

		return QString();
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

	ViewType
	getViewType()const
	{
		return getViewerState().viewType;
	}

	QStringList 
	getAvailableColorTransformations()
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
			strList << QString::fromStdWString( (*idList)[ i ].name );
		}

		return strList;
	}
	
	void
	setRenderingExtension( RenderingExtension::Ptr aRenderingExtension )
	{
		mRenderingExtension = aRenderingExtension;
		update();
	}
public slots:
	void
	setViewType( int aViewType )
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


		update();

		emit ViewTypeChanged( aViewType );
	}

	void
	setColorTransformType( const QString & aColorTransformName )
	{
		//TODO
		std::wstring name = aColorTransformName.toStdWString();
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

	void
	zoomFit( ZoomType zoomType = ztFIT );

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

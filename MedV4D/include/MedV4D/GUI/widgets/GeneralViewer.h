#ifndef GENERAL_VIEWER_H
#define GENERAL_VIEWER_H

#include "MedV4D/Imaging/Imaging.h"

#include "MedV4D/GUI/utils/ViewConfiguration.h"
#include "MedV4D/GUI/utils/IUserEvents.h"
#include "MedV4D/GUI/utils/TransferFunctionBuffer.h"
#include "MedV4D/GUI/utils/CgShaderTools.h"
#include "MedV4D/GUI/utils/GLTextureImage.h"
#include "MedV4D/GUI/utils/FrameBufferObject.h"
#include "MedV4D/GUI/utils/ViewerController.h"
#include "MedV4D/GUI/utils/MouseTracking.h"
#include "MedV4D/GUI/widgets/AGUIViewer.h"
#include "MedV4D/GUI/widgets/ViewerConstructionKit.h"

#include "MedV4D/GUI/widgets/AGLViewer.h"
#include "MedV4D/GUI/renderers/RendererTools.h"
#include "MedV4D/GUI/renderers/SliceRenderer.h"
#include "MedV4D/GUI/renderers/VolumeRenderer.h"

#include <QtGui>
//#include <QtOpenGL>
#include <boost/shared_ptr.hpp>
#include <boost/cast.hpp>
#include <map>

#include "MedV4D/Common/Types.h"



namespace M4D
{
namespace GUI
{
namespace Viewer
{

class GeneralViewer;
enum QualityMode {
		qmLow = 0,
		qmNormal = 1,
		qmHigh = 2,
		qmFinest = 3
	};

class ViewerState : public BaseViewerState
{
public:
	typedef boost::shared_ptr< ViewerState > Ptr;
	
	//GLTextureImage::Ptr	_textureData;

	unsigned colorTransform;
	
	Vector3i
	getMaxSlice()const
	{
		return mPrimaryImageExtents.maximum;
	}

	Vector3i
	getMinSlice()const
	{
		return mPrimaryImageExtents.minimum;
	}
	
	Vector3f
	getRealSize()const
	{
		return mPrimaryImageExtents.realMaximum - mPrimaryImageExtents.realMinimum;
	}
	
	Vector3f
	getRealCenter()const
	{
		return 0.5f * (mPrimaryImageExtents.realMaximum + mPrimaryImageExtents.realMinimum);
	}
	Vector3f
	getMinimalElementExtents()const
	{
		return mPrimaryImageExtents.elementExtents;
	}
	
	M4D::Imaging::ImageExtentsRecord<3> mPrimaryImageExtents;
	GLTextureImage::WPtr mPrimaryImageTexture;
	M4D::Common::TimeStamp mPrimaryEditTimestamp;
	
	M4D::Imaging::ImageExtentsRecord<3> mSecondaryImageExtents;
	GLTextureImage::WPtr mSecondaryImageTexture;
	M4D::Common::TimeStamp mSecondaryEditTimestamp;
	
	/*Vector< float, 3 > 			_regionRealMin;
	Vector< float, 3 >			_regionRealMax;
	Vector< float, 3 >			_elementExtents;
	Vector< int32, 3 > 			_regionMin;
	Vector< int32, 3 >			_regionMax;*/

	//TransferFunctionBuffer1D::Ptr 		mTFunctionBuffer;
	//GLTransferFunctionBuffer1D::Ptr 	mTransferFunctionTexture;
	TransferFunctionBufferInfo		mTransferFunctionBufferInfo;

	M4D::GUI::Renderer::SliceRenderer	mSliceRenderer;
	M4D::GUI::Renderer::SliceRenderer::RenderingConfiguration mSliceRenderConfig;
	Vector2u				m2DMultiSliceGrid;
	size_t					m2DMultiSliceStep;

	M4D::GUI::Renderer::VolumeRenderer	mVolumeRenderer;
	M4D::GUI::Renderer::VolumeRenderer::RenderingConfiguration mVolumeRenderConfig;
	bool 					mEnableVolumeBoundingBox;
	QualityMode				mQualityMode;
	
	
	CgEffect mSceneSlicingCgEffect;
	CgEffect mBasicCgEffect;
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
					PortInterfaceHelper< boost::mpl::vector< M4D::Imaging::AImage, M4D::Imaging::AImage > >
					>
{
	Q_OBJECT;
public:
	typedef ViewerConstructionKit<  AGLViewer, 
					PortInterfaceHelper< boost::mpl::vector< M4D::Imaging::AImage, M4D::Imaging::AImage > >
					>	PredecessorType;


	GeneralViewer( QWidget *parent = NULL );

	void
	setTiling( unsigned aRows, unsigned aCols );
	
	void
	setTiling( unsigned aRows, unsigned aCols, unsigned aSliceStep );
	
	Vector2u
	getTiling() const;
	
	size_t
	getTilingSliceStep() const;

	void
	setLUTWindow( float32 center, float32 width );

	void
	setLUTWindow( Vector2f window );

	Vector2f
	getLUTWindow() const;

	void
	setTransferFunctionBuffer( TransferFunctionBuffer1D::Ptr aTFunctionBuffer );

	void
	setTransferFunctionBufferInfo( TransferFunctionBufferInfo aTFunctionBufferInfo );

	TransferFunctionBufferInfo
	getTransferFunctionBufferInfo()const;

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

	float32
	getCurrentRealSlice()const;

	void
	changeCurrentSlice( int32 diff );
	
	bool
	isColorTransformAvailable( unsigned aTransformType );

	void
	setVolumeRestrictions( const Vector2f &aX, const Vector2f &aY, const Vector2f &aZ );

	void
	setVolumeRestrictions( bool aEnable, const Vector2f &aX, const Vector2f &aY, const Vector2f &aZ );

	void
	setCutPlane( const Planef &aCutPlane );

	Planef
	getCutPlane()const;

	void
	setCutPlaneCameraTargetOffset( float aOffset );

	void
	getVolumeRestrictions( Vector2f &aX, Vector2f &aY, Vector2f &aZ )const;

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

	bool
	isIntegratedTransferFunctionEnabled() const;

	bool
	isVolumeRestrictionEnabled() const;

	bool
	isCutPlaneEnabled() const;

	bool
	isInterpolationEnabled() const;

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
	
	bool
	isBoundingBoxEnabled()const;

	QualityMode
	getRenderingQuality();

	Vector3f
	getCameraPosition()const
	{ return getViewerState().mVolumeRenderConfig.camera.GetEyePosition(); }

	Vector3f
	getCameraTargetPosition()const
	{ return getViewerState().mVolumeRenderConfig.camera.GetTargetPosition(); }

	Vector3f
	getCameraTargetDirection()const
	{ return getViewerState().mVolumeRenderConfig.camera.GetTargetDirection(); }

	void
	setSliceCountForRenderingQualities( int aLow, int aNormal, int aHigh, int aFinest );

	void
	updateMouseInfo( Vector3f aDataCoords );

	QString
	GetVoxelInfo( Vector3f aDataCoords );

	QStringList
	getPredefinedZoomValueNames()const;

	QString
	getZoomValueName()const;


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
	enableIntegratedTransferFunction( bool aEnable );

	void
	setJitterStrength( float aValue );

	void
	enableVolumeRestrictions( bool aEnable );

	void
	enableCutPlane( bool aEnable );

	void
	setRenderingQuality( int aQualityMode );

	void
	enableBoundingBox( bool aEnable );

	void
	enableInterpolation( bool aEnable );
	
	void
	resetView();

	void
	zoomFit( ZoomType zoomType = ztFIT );

	void
	setZoom( ZoomType zoomType );

	void
	setZoom( float aZoom );

	void
	setZoom( const QString &aZoom );

signals:
	void
	settingsChanged();

	void
	ViewTypeChanged( int aRendererType );

	void
	ColorTransformTypeChanged( int aColorTransform );

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

	MouseEventInfo
	getMouseEventInfo( QMouseEvent * event );

	void	
	resizeGL ( int width, int height )
	{
		PredecessorType::resizeGL( width, height );
		zoomFit();
	}

//******** TMP ************
	bool
	IsDataPrepared();

	bool
	PrepareData();

	

	bool _prepared;
//******** TMP ************

	RenderingExtension::Ptr mRenderingExtension;

	Vector4i mSliceCountForRenderingQualities;

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

#ifndef GENERAL_VIEWER_H
#define GENERAL_VIEWER_H
//Temporary workaround
#ifndef Q_MOC_RUN

#include "MedV4D/Imaging/Imaging.h"
#include "MedV4D/GUI/utils/ViewerController.h"

#include <soglu/GLSLShader.hpp>
#include <soglu/GLTextureImage.hpp>
#include "MedV4D/GUI/widgets/ViewerConstructionKit.h"
#include "MedV4D/GUI/widgets/AGLViewer.h"



///#include "MedV4D/GUI/utils/ViewConfiguration.h"
//#include "MedV4D/GUI/utils/IUserEvents.h"

/*#include "MedV4D/GUI/utils/MouseTracking.h"
#include "MedV4D/GUI/widgets/AGUIViewer.h"

*/

//#include <vorgl/TransferFunctionBuffer.hpp>
//#include "MedV4D/GUI/utils/IDMappingBuffer.h"

#include "MedV4D/GUI/renderers/RendererTools.h"
#include "MedV4D/GUI/renderers/SliceRenderer.h"
#include "MedV4D/GUI/renderers/VolumeRenderer.h"

#include <QtWidgets>
//#include <QtOpenGL>
#include <memory>
#include <boost/cast.hpp>
#include <map>

#include "MedV4D/Common/Types.h"

#endif

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
	typedef std::shared_ptr< ViewerState > Ptr;
	Vector3i
	getMaxSlice()const { return mPrimaryImageExtents.maximum; }
	Vector3i
	getMinSlice()const { return mPrimaryImageExtents.minimum; }
	Vector3f
	getRealSize()const { return mPrimaryImageExtents.realMaximum - mPrimaryImageExtents.realMinimum; }
	Vector3f
	getRealCenter()const { return 0.5f * (mPrimaryImageExtents.realMaximum + mPrimaryImageExtents.realMinimum); }
	Vector3f
	getMinimalElementExtents()const {	return mPrimaryImageExtents.elementExtents; }

	M4D::Imaging::ImageExtentsRecord<3> mPrimaryImageExtents;
	soglu::GLTextureImage::WPtr mPrimaryImageTexture;
	M4D::Common::TimeStamp mPrimaryEditTimestamp;

	M4D::Imaging::ImageExtentsRecord<3> mSecondaryImageExtents;
	soglu::GLTextureImage::WPtr mSecondaryImageTexture;
	M4D::Common::TimeStamp mSecondaryEditTimestamp;

	vorgl::TransferFunctionBufferInfo		mTransferFunctionBufferInfo;

//	IDMappingBufferInfo			mMappingBufferInfo;

	unsigned colorTransform;

	M4D::GUI::Renderer::SliceRenderer	mSliceRenderer;
	M4D::GUI::Renderer::SliceRenderer::RenderingConfiguration mSliceRenderConfig;
	Vector2i				m2DMultiSliceGrid;
	int					m2DMultiSliceStep;

	M4D::GUI::Renderer::VolumeRenderer	mVolumeRenderer;
	M4D::GUI::Renderer::VolumeRenderer::RenderingConfiguration mVolumeRenderConfig;
	bool 					mEnableVolumeBoundingBox;
	QualityMode				mQualityMode;
};

class RenderingExtension
{
public:
	typedef std::shared_ptr< RenderingExtension > Ptr;

	virtual unsigned
	getAvailableViewTypes()const = 0;

	virtual void
	render2DAlignedSlices( int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane ) = 0;

	virtual void
	preRender3D() = 0;

	virtual void
	postRender3D() = 0;
};


class ViewerInputData
{
public:
	typedef std::shared_ptr<ViewerInputData> Ptr;
	typedef std::shared_ptr<const ViewerInputData> ConstPtr;

	ViewerInputData()
	{}

	ViewerInputData(M4D::Imaging::AImageDim<3>::ConstPtr aPrimaryImage, M4D::Imaging::AImageDim<3>::ConstPtr aSecondaryImage = M4D::Imaging::AImageDim<3>::ConstPtr())
		: mPrimaryImage(aPrimaryImage)
		, mSecondaryImage(aSecondaryImage)
	{}

	M4D::Imaging::AImageDim<3>::ConstPtr
	primaryImage() const
	{
		return mPrimaryImage;
	}

	M4D::Imaging::AImageDim<3>::ConstPtr
	secondaryImage() const
	{
		return mSecondaryImage;
	}

protected:
	M4D::Imaging::AImageDim<3>::ConstPtr mPrimaryImage;
	M4D::Imaging::AImageDim<3>::ConstPtr mSecondaryImage;
};


class GeneralViewer : public AGLViewer
	/*public ViewerConstructionKit<   AGLViewer,
					PortInterfaceHelper< boost::mpl::vector< M4D::Imaging::AImage, M4D::Imaging::AImage > >
					>*/
{
	Q_OBJECT;
public:
	/*typedef ViewerConstructionKit<  AGLViewer,
					PortInterfaceHelper< boost::mpl::vector< M4D::Imaging::AImage, M4D::Imaging::AImage > >
					>	PredecessorType;*/
	typedef AGLViewer PredecessorType;

	GeneralViewer(QWidget *parent = nullptr);

	void
	setTiling(int aRows, int aCols);

	void
	setTiling(int aRows, int aCols, int aSliceStep);

	Vector2i
	getTiling() const;

	size_t
	getTilingSliceStep() const;

	void
	setLUTWindow( float32 center, float32 width );

	void
	setLUTWindow(glm::fvec2 window);

	glm::fvec2
	getLUTWindow() const;

	void
	setTransferFunctionBuffer( vorgl::TransferFunctionBuffer1D::Ptr aTFunctionBuffer );

	void
	setTransferFunctionBufferInfo( vorgl::TransferFunctionBufferInfo aTFunctionBufferInfo );

	vorgl::TransferFunctionBufferInfo
	getTransferFunctionBufferInfo()const;

	/*void
	setMappingBuffer( IDMappingBuffer::Ptr aMappingBuffer );*/

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
	setCutPlane( const soglu::Planef &aCutPlane );

	soglu::Planef
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

	glm::fvec3
	getCameraPosition()const
	{ return getViewerState().mVolumeRenderConfig.renderingConfiguration.camera.eyePosition(); }

	glm::fvec3
	getCameraTargetPosition()const
	{ return getViewerState().mVolumeRenderConfig.renderingConfiguration.camera.targetPosition(); }

	glm::fvec3
	getCameraTargetDirection()const
	{ return getViewerState().mVolumeRenderConfig.renderingConfiguration.camera.targetDirection(); }

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

	float
	isoSurfaceValue() const {
		return getViewerState().mVolumeRenderConfig.isoSurfaceOptions.isoValue;
	}

	void
	setIsoSurfaceValue(float aValue) {
		getViewerState().mVolumeRenderConfig.isoSurfaceOptions.isoValue = aValue;
		emit settingsChanged();
	}

	QColor
	isoSurfaceColor() const
	{
		const auto &c = getViewerState().mVolumeRenderConfig.isoSurfaceOptions.isoSurfaceColor;
		return QColor::fromRgbF(c[0], c[1], c[2], c[3]);
	}

	void
	setIsoSurfaceColor(QColor aColor) {
		getViewerState().mVolumeRenderConfig.isoSurfaceOptions.isoSurfaceColor =
			glm::fvec4(
				aColor.redF(),
				aColor.greenF(),
				aColor.blueF(),
				aColor.alphaF()
				);
		emit settingsChanged();
	}

	void
	setInputData(ViewerInputData::ConstPtr aData);

	ViewerInputData::ConstPtr
	inputData() const
	{
		return mData;
	}

//public slots:
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

	void
	reloadShaders();

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

	ViewerInputData::ConstPtr mData;

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

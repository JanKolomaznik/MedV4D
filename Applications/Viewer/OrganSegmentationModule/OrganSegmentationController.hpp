#ifndef ORGAN_SEGMENTATION_CONTROLLER_H
#define ORGAN_SEGMENTATION_CONTROLLER_H

#include <QtCore>
#include "MedV4D/GUI/widgets/GeneralViewer.h"
#include "MedV4D/GUI/managers/ApplicationManager.h"
#include "MedV4D/GUI/utils/ProxyViewerController.h"
#include "MedV4D/GUI/utils/QtModelViewTools.h"
#include "MedV4D/GUI/utils/DrawingMouseController.h"
#include "MedV4D/GUI/utils/PrimitiveCreationEventController.h"
#include "MedV4D/GUI/utils/IDMappingBuffer.h"
#include "MedV4D/GUI/managers/OpenGLManager.h"

#include "MedV4D/Imaging/painting/Paint.h"
#include "MedV4D/Imaging/CanonicalProbModel.h"
#include <boost/function.hpp>

enum class BrushType {
	Circle = 0,
	Sphere,
	Square,
	Cube
};

struct DrawingBrush
{
	BrushType brushType;
	double radius;
};

class MaskDrawingMouseController: public ADrawingMouseController
{
public:
	typedef std::shared_ptr<MaskDrawingMouseController> Ptr;
	MaskDrawingMouseController(M4D::Imaging::Mask3D::Ptr aMask = M4D::Imaging::Mask3D::Ptr(), uint8 aBrushValue = 255)
		: mMask( aMask )
		, mBrushValue( aBrushValue )
	{
	}

	void
	setMask(M4D::Imaging::Mask3D::Ptr aMask)
	{
		mMask = aMask;
	}

	void
	setBrushValue( uint8 aBrushValue )
	{
		mBrushValue = aBrushValue;
	}

	void
	setBrush(DrawingBrush aBrush)
	{
		mBrush = aBrush;
	}

protected:
	void
	drawStep(const Vector3f &aStart, const Vector3f &aEnd, const Vector3f &aNormal) override
	{
		drawMaskStep(aStart, aEnd, aNormal);
	}

	void
	drawMaskStep( const Vector3f &aStart, const Vector3f &aEnd, const Vector3f &aNormal)
	{
		ASSERT( mMask );
		float width = mBrush.radius;
		Vector3f offset( width, width, width );
		Vector3f minimum = M4D::minVect<float,3>( M4D::minVect<float,3>( aStart - offset, aStart + offset ), M4D::minVect<float,3>( aEnd - offset, aEnd + offset ) );
		Vector3f maximum = M4D::maxVect<float,3>( M4D::maxVect<float,3>( aStart - offset, aStart + offset ), M4D::maxVect<float,3>( aEnd - offset, aEnd + offset ) );

		Vector3i c1 = M4D::maxVect<int,3>( mMask->GetElementCoordsFromWorldCoords( minimum ), mMask->GetMinimum() );
		Vector3i c2 = M4D::minVect<int,3>( mMask->GetElementCoordsFromWorldCoords( maximum ), mMask->GetMaximum() );
		M4D::Imaging::WriterBBoxInterface & mod = mMask->SetDirtyBBox( c1, c2 );
		LOG( "EditedA : " << c1 << " => " << c2 );
		try {
			switch (mBrush.brushType) {
			case BrushType::Square:
			case BrushType::Circle:
			case BrushType::Cube:
			case BrushType::Sphere:
				M4D::Imaging::painting::draw3DLine(*mMask, mBrushValue, aStart, aEnd, mBrush.radius, aNormal);
				break;
			default:
				assert(false);
			}
			//M4D::Imaging::painting::drawRectangleAlongLine( *mMask, mBrushValue, aStart, aEnd, 10, aNormal);
		} catch (...){
			D_PRINT( "drawStep exception" );
		}
		//M4D::Imaging::WriterBBoxInterface & mod = mMask->SetWholeDirtyBBox();
		mod.SetModified();
		OpenGLManager::getInstance()->getTextureFromImage(*mMask);
		/*D_PRINT( "DRAW " << aStart );
		Vector3f diff = 0.1f*(aEnd-aStart);
		for( size_t i = 0; i <= 10; ++i ) {
			try {
				mMask->GetElementWorldCoords( aStart + float(i) * diff ) = 255;
			} catch (...){
				D_PRINT( "drawStep exception" );
			}
		}*/
	}

	M4D::Imaging::Mask3D::Ptr mMask;
	uint8 mBrushValue;
	DrawingBrush mBrush;
};

class RegionMarkingMouseController: public ADrawingMouseController
{
public:
	typedef std::shared_ptr<RegionMarkingMouseController> Ptr;
	typedef std::set< uint32 > ValuesSet;

	RegionMarkingMouseController(
			M4D::Imaging::Image3DUnsigned32b::Ptr aRegions,
			M4D::GUI::IDMappingBuffer::Ptr aIDMappingBuffer,
			std::shared_ptr< ValuesSet > aValues,
			std::function<void ()> aUpdateCallback
			)
		: mRegions( aRegions )
		, mIDMappingBuffer( aIDMappingBuffer )
		, mValues( aValues )
		, mUpdateCallback( aUpdateCallback )
	{
		ASSERT( mRegions ); }

	void
	setValuesSet( std::shared_ptr< ValuesSet > aValues )
	{
		mValues = aValues;
	}

	void
	setRegions( M4D::Imaging::Image3DUnsigned32b::Ptr aRegions )
	{
		mRegions = aRegions;
	}
protected:
	void
	drawStep( const Vector3f &aStart, const Vector3f &aEnd, const Vector3f &aNormal) override
	{
		ASSERT( mValues );
		ASSERT( mRegions );
		drawMaskStep( aStart, aEnd );
	}

	void
	drawMaskStep( const Vector3f &aStart, const Vector3f &aEnd )
	{
		float width = 10.0f;
		Vector3f offset( width, width, width );
//		Vector3f minimum = M4D::minVect<float,3>( M4D::minVect<float,3>( aStart - offset, aStart + offset ), M4D::minVect<float,3>( aEnd - offset, aEnd + offset ) );
//		Vector3f maximum = M4D::maxVect<float,3>( M4D::maxVect<float,3>( aStart - offset, aStart + offset ), M4D::maxVect<float,3>( aEnd - offset, aEnd + offset ) );

		//Vector3i c1 = M4D::maxVect<int,3>( mRegions->GetElementCoordsFromWorldCoords( minimum ), mRegions->GetMinimum() );
		//Vector3i c2 = M4D::minVect<int,3>( mRegions->GetElementCoordsFromWorldCoords( maximum ), mRegions->GetMaximum() );
		try {
			M4D::Imaging::painting::getValuesFromRectangleAlongLine( *mRegions, *mValues, aStart, aEnd, 10, Vector3f( 0.0f, 0.0f, 1.0f ) );
			//M4D::Imaging::painting::drawRectangleAlongLine( *mMask, mBrushValue, aStart, aEnd, 10, Vector3f( 0.0f, 0.0f, 1.0f ) );
		} catch (std::exception &e){
			D_PRINT( "drawStep exception: " << e.what() );
		}

		mUpdateCallback();
	}

	M4D::Imaging::Image3DUnsigned32b::Ptr mRegions;
	M4D::GUI::IDMappingBuffer::Ptr mIDMappingBuffer;
	std::shared_ptr< ValuesSet > mValues;
	std::function<void ()> mUpdateCallback;
};


class OrganSegmentationModule;

class OrganSegmentationController: public ModeProxyViewerController, public M4D::GUI::Viewer::RenderingExtension
{
	Q_OBJECT;
public:
	enum class MarkerType {
		none,
		foreground,
		background
	};

	typedef QList<QAction *> QActionList;
	typedef std::shared_ptr< OrganSegmentationController > Ptr;
	typedef M4D::GUI::Viewer::ViewerController ControllerPredecessor;
	typedef M4D::GUI::Viewer::RenderingExtension RenderingExtPredecessor;

	OrganSegmentationController( OrganSegmentationModule &aModule );
	~OrganSegmentationController();

	void
	activated()
	{

	}

	void
	deactivated()
	{
		//std::for_each( mChosenToolActions.begin(), mChosenToolActions.end(), boost::bind( &QAction::setChecked, _1, false ) );
	}

	void
	setModeId( M4D::Common::IDNumber aId )
	{
		mModeId = aId;
	}

	/*bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo );

	bool
	mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo );

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo );

	bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo );

	bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event );*/

	unsigned
	getAvailableViewTypes()const;

	void
	render2DAlignedSlices( int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane );

	void
	preRender3D();

	void
	postRender3D();

	void
	render3D();

	M4D::Imaging::Mask3D::Ptr	mMask;

	void
	setIDMappingBuffer( M4D::GUI::IDMappingBuffer::Ptr aIDMappingBuffer )
	{ mIDMappingBuffer = aIDMappingBuffer; }

	void
	changeMarkerType(MarkerType aMarkerType);
signals:
	void
	updateRequest();

public slots:
	void
	toggleMaskDrawing( bool aToggle );

	void
	toggleRegionMarking( bool aToggle );

	void
	toggleBiMaskDrawing( bool aToggle );
protected:

public:
	M4D::GUI::IDMappingBuffer::Ptr mIDMappingBuffer;

	Qt::MouseButton	mVectorEditorInteractionButton;

	bool mOverlay;

	M4D::Common::IDNumber mModeId;

	OrganSegmentationModule &mModule;

	//ADrawingMouseController::Ptr mMaskDrawingController;
	MaskDrawingMouseController::Ptr mMaskDrawingController;
	MarkerType mMarkerType;

	uint8 mBrushValue;
};



#endif /*ORGAN_SEGMENTATION_CONTROLLER_H*/

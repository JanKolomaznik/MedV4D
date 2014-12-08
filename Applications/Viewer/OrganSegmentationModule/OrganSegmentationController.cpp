#include "OrganSegmentationModule/OrganSegmentationController.hpp"
#include "OrganSegmentationModule/OrganSegmentationModule.hpp"
#include <algorithm>

OrganSegmentationController::OrganSegmentationController( OrganSegmentationModule &aModule )
	: mModule( aModule )
	, mMarkerType(MarkerType::foreground)
	, mBrushValue( 255 )
{
	mMaskDrawingController = MaskDrawingMouseController::Ptr(new MaskDrawingMouseController());

}

OrganSegmentationController::~OrganSegmentationController()
{

}

unsigned
OrganSegmentationController::getAvailableViewTypes()const
{
	return 0;
}

void
OrganSegmentationController::render2DAlignedSlices( int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane )
{

}

void
OrganSegmentationController::preRender3D()
{
	if( !mOverlay ) {
		render3D();
	}
}

void
OrganSegmentationController::postRender3D()
{
	if( mOverlay ) {
		render3D();
	}
}

void
OrganSegmentationController::render3D()
{

}

void
OrganSegmentationController::toggleMaskDrawing( bool aToggle )
{
	if( aToggle ) {
		mMaskDrawingController->setMask(mMask);
		changeMarkerType(MarkerType::foreground);
		setController(mMaskDrawingController);
	} else {
		resetController();//mMaskDrawingController.reset();
	}
}

void
OrganSegmentationController::toggleRegionMarking( bool aToggle )
{
	if( aToggle ) {
		ASSERT(mModule.getGraphCutSegmentationWrapper().mWatersheds);
		setController(std::make_shared<RegionMarkingMouseController>(
					mModule.getGraphCutSegmentationWrapper().mWatersheds,
					mIDMappingBuffer,
					mModule.getGraphCutSegmentationWrapper().mForegroundMarkers,
					boost::bind( &OrganSegmentationModule::update, mModule )
					)
				);
		D_PRINT( "Switched to region marking controller" );
	} else {
		resetController();//mMaskDrawingController.reset();
	}
}

void
OrganSegmentationController::toggleBiMaskDrawing( bool aToggle )
{
	if( aToggle ) {
		//setController(MaskDrawingMouseController::Ptr(new MaskDrawingMouseController(mMask, 255)));
		mMaskDrawingController->setMask(mMask);
		setController(mMaskDrawingController);
	} else {
		resetController();
	}
	/*if( aToggle ) {
		mMaskDrawingController = RegionMarkingMouseController::Ptr( new RegionMarkingMouseController( mModule.getGraphCutSegmentationWrapper().mWatersheds, mModule.getGraphCutSegmentationWrapper().mForegroundMarkers ) );
	} else {
		mMaskDrawingController.reset();
	}*/
}

void
OrganSegmentationController::changeMarkerType(MarkerType aMarkerType)
{
	mMarkerType = aMarkerType;
	switch (mMarkerType) {
	case MarkerType::background :
		mMaskDrawingController->setBrushValue(128);
		break;
	case MarkerType::foreground :
		mMaskDrawingController->setBrushValue(255);
		break;
	case MarkerType::none :
		mMaskDrawingController->setBrushValue(0);
		break;
	default:
		ASSERT(false);
	}

	/*if (mMaskDrawingController && dynamic_cast<RegionMarkingMouseController *>( mMaskDrawingController.get()) ) {
		RegionMarkingMouseController &controller = dynamic_cast<RegionMarkingMouseController &>(*mMaskDrawingController);
		if ( aForeground ) {
			controller.setValuesSet( mModule.getGraphCutSegmentationWrapper().mForegroundMarkers );
		} else {
			controller.setValuesSet( mModule.getGraphCutSegmentationWrapper().mBackgroundMarkers );
		}
	}*/
	/*mBrushValue = aForeground ? 255 : 100;
	if( mMaskDrawingController && boost::dynamic_pointer_cast<>( mMaskDrawingController ) ) {
			mMaskDrawingController->setBrushValue( mBrushValue );
	}*/
}




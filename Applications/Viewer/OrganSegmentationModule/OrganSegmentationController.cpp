#include "OrganSegmentationModule/OrganSegmentationController.hpp"
#include "OrganSegmentationModule/OrganSegmentationModule.hpp"
#include <algorithm>

OrganSegmentationController::OrganSegmentationController( OrganSegmentationModule &aModule ): mModule( aModule ), mBrushValue( 255 )
{

	
}

OrganSegmentationController::~OrganSegmentationController()
{

}

unsigned
OrganSegmentationController::getAvailableViewTypes()const
{

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
		setController( MaskDrawingMouseController::Ptr( new MaskDrawingMouseController( mMask, 255 ) ) );
	} else {
		resetController();//mMaskDrawingController.reset();
	}
}

void
OrganSegmentationController::toggleRegionMarking( bool aToggle )
{
	if( aToggle ) {
		ASSERT(mModule.getGraphCutSegmentationWrapper().mWatersheds);
		setController( RegionMarkingMouseController::Ptr( new RegionMarkingMouseController( mModule.getGraphCutSegmentationWrapper().mWatersheds, mModule.getGraphCutSegmentationWrapper().mForegroundMarkers ) ) );
	} else {
		resetController();//mMaskDrawingController.reset();
	}
}

void
OrganSegmentationController::toggleBiMaskDrawing( bool aToggle )
{
	/*if( aToggle ) {
		mMaskDrawingController = RegionMarkingMouseController::Ptr( new RegionMarkingMouseController( mModule.getGraphCutSegmentationWrapper().mWatersheds, mModule.getGraphCutSegmentationWrapper().mForegroundMarkers ) );
	} else {
		mMaskDrawingController.reset();
	}*/
}

void
OrganSegmentationController::changeMarkerType( bool aForeground )
{	
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




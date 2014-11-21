#include "OrganSegmentationModule/OrganSegmentationWidget.hpp"
#include "OrganSegmentationModule/OrganSegmentationModule.hpp"


void
OrganSegmentationWidget::createMask()
{
	mModule.createMask();
}

void
OrganSegmentationWidget::loadMask()
{
	mModule.loadMask();
}

void
OrganSegmentationWidget::loadModel()
{
	mModule.loadModel();
}

void
OrganSegmentationWidget::computeStats()
{
	mModule.computeStats();
}

void
OrganSegmentationWidget::loadIndex()
{
	mModule.loadIndexFile();
}

void
OrganSegmentationWidget::toggleDraw( bool aToggle )
{
	mController->toggleMaskDrawing( aToggle );
}

void
OrganSegmentationWidget::toggleBiMaskDraw( bool aToggle )
{
	changedMarkerType();
	mController->toggleBiMaskDrawing( aToggle );
}

void
OrganSegmentationWidget::changedMarkerType()
{
	if (mForegroundRadioButton->isChecked()) {
		mController->changeMarkerType(OrganSegmentationController::MarkerType::foreground);
	}
	if (mBackgroundRadioButton->isChecked()) {
		mController->changeMarkerType(OrganSegmentationController::MarkerType::background);
	}
}

void
OrganSegmentationWidget::updateTimestamp()
{
	mModule.updateTimestamp();
}

void
OrganSegmentationWidget::runSegmentation()
{
	mModule.computeSegmentation();
}

void
OrganSegmentationWidget::buttonPressed()
{
	STUBBED("Remove (probably)");
	/*QObject * obj = sender();
	if (obj == buttonComputeWatersheds) {
		mModule.computeWatershedTransformation();
		return;
	}
	if (obj == buttonRunSegmentation) {
		mModule.computeSegmentation();
		return;
	}*/

}

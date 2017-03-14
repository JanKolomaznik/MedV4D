#include "OrganSegmentationModule/OrganSegmentationWidget.hpp"
#include "OrganSegmentationModule/OrganSegmentationModule.hpp"


OrganSegmentationWidget::OrganSegmentationWidget(OrganSegmentationController::Ptr aController, OrganSegmentationModule &aModule)
	: mController( aController )
	, mModule( aModule )
{
	ASSERT( aController );
	setupUi( this );

	QObject::connect(mBrushSettings, &BrushSettingsForm::updated, this, &OrganSegmentationWidget::brushChanged);

	brushChanged();
}

void
OrganSegmentationWidget::createMask()
{
	mModule.createMask();
}

void OrganSegmentationWidget::clearMask()
{
	mModule.clearMask();
}

void OrganSegmentationWidget::fillMaskBorder()
{
	mModule.fillMaskBorder(mRadiusPercentageSpinBox->value());
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
	if (mEraseRadioButton->isChecked()) {
		mController->changeMarkerType(OrganSegmentationController::MarkerType::none);
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

void OrganSegmentationWidget::brushChanged()
{
	mController->mMaskDrawingController->setBrush(mBrushSettings->brush());
}

void OrganSegmentationWidget::varianceUpdated()
{
	mModule.setVariance(float(mVarianceSpinBox->value()));
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

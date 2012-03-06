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
OrganSegmentationWidget::updateTimestamp()
{
	mModule.updateTimestamp();
}
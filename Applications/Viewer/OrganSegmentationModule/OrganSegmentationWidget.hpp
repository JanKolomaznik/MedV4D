#ifndef ORGAN_SEGMENTATION_WIDGET_HPP
#define ORGAN_SEGMENTATION_WIDGET_HPP

#include "MedV4D/Common/Common.h"
#include "ui_OrganSegmentationWidget.h"
#include "OrganSegmentationModule/OrganSegmentationController.hpp"
#include <QtGui>
#include <iostream>
#include <fstream>

class OrganSegmentationModule;

class OrganSegmentationWidget: public QWidget, public Ui::OrganSegmentationWidget
{
	Q_OBJECT;
public:
	OrganSegmentationWidget( OrganSegmentationController::Ptr aController, OrganSegmentationModule &aModule );
public slots:
	void
	createMask();
	void
	clearMask();

	void
	fillMaskBorder();

	void
	loadMask();
	void
	loadModel();
	void
	computeStats();
	void
	loadIndex();
	void
	toggleDraw( bool aToggle );
	void
	toggleBiMaskDraw( bool aToggle );
	void
	changedMarkerType();
	void
	updateTimestamp();

	//void
	//buttonPressed();

	void
	runSegmentation();

	void
	brushChanged();

	void
	varianceUpdated();

	void
	watershedTransformation();
protected:
	OrganSegmentationController::Ptr mController;
	OrganSegmentationModule &mModule;
};


#endif /*ORGAN_SEGMENTATION_WIDGET_HPP*/

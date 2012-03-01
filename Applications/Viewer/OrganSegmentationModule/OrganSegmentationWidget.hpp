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
	OrganSegmentationWidget( OrganSegmentationController::Ptr aController, OrganSegmentationModule &aModule ): mController( aController ), mModule( aModule )
	{
		ASSERT( aController );
		setupUi( this );
	}
public slots:
	void
	createMask();
	void
	loadMask();
	void
	loadModel();
	void
	toggleDraw( bool aToggle );
	void
	updateTimestamp();

protected:
	OrganSegmentationController::Ptr mController;
	OrganSegmentationModule &mModule;
};


#endif /*ORGAN_SEGMENTATION_WIDGET_HPP*/

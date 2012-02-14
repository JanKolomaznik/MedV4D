#ifndef ORGAN_SEGMENTATION_WIDGET_HPP
#define ORGAN_SEGMENTATION_WIDGET_HPP

#include "MedV4D/Common/Common.h"
#include "ui_OrganSegmentationWidget.h"
#include "OrganSegmentationModule/OrganSegmentationController.hpp"
#include <QtGui>
#include <iostream>
#include <fstream>

class OrganSegmentationWidget: public QWidget, public Ui::OrganSegmentationWidget
{
	Q_OBJECT;
public:
	OrganSegmentationWidget( OrganSegmentationController::Ptr aController ): mController( aController )
	{
		ASSERT( aController );
		setupUi( this );
	}
public slots:

protected:
	OrganSegmentationController::Ptr mController;
};


#endif /*ORGAN_SEGMENTATION_WIDGET_HPP*/

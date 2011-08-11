#ifndef SHOULDER_MEASUREMENT_WIDGET_HPP
#define SHOULDER_MEASUREMENT_WIDGET_HPP

#include "common/Common.h"
#include "ui_ShoulderMeasurementWidget.h"
#include "ShoulderMeasurementModule/ShoulderMeasurementController.hpp"
#include <QtGui>
#include <iostream>
#include <fstream>

class ShoulderMeasurementWidget: public QWidget, public Ui::ShoulderMeasurementWidget
{
	Q_OBJECT;
public:
	ShoulderMeasurementWidget( ShoulderMeasurementController::Ptr aController ): mController( aController )
	{
		ASSERT( aController );
		setupUi( this );

		pointsView->setModel( &(aController->getPointModel()) );
	}
public slots:
	void
	startHumeralHeadPointDefining( bool aStart )
	{
		if( aStart ) {
			mController->setMeasurementMode( ShoulderMeasurementController::mmHUMERAL_HEAD );
		} else {
			mController->setMeasurementMode( ShoulderMeasurementController::mmNONE );
		}	
	}
protected:
	ShoulderMeasurementController::Ptr mController;
};


#endif /*SHOULDER_MEASUREMENT_WIDGET_HPP*/

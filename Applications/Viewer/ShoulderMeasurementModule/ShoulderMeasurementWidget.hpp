#ifndef SHOULDER_MEASUREMENT_WIDGET_HPP
#define SHOULDER_MEASUREMENT_WIDGET_HPP

#include "MedV4D/Common/Common.h"
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

		proximalShaftPointsView->setModel( &(aController->getProximalShaftPointModel()) );
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

	void
	startProximalShaftPointDefining( bool aStart )
	{
		if( aStart ) {
			mController->setMeasurementMode( ShoulderMeasurementController::mmPROXIMAL_SHAFT );
		} else {
			mController->setMeasurementMode( ShoulderMeasurementController::mmNONE );
		}	
	}

	void
	analyseHumeralHead()
	{
		mController->analyseHumeralHead();
	}

	void
	analyseProximalShaftOfHumerus()
	{
		mController->analyseProximalShaftOfHumerus();
	}
protected:
	ShoulderMeasurementController::Ptr mController;
};


#endif /*SHOULDER_MEASUREMENT_WIDGET_HPP*/

#ifndef MEASUREMENT_WIDGET_HPP
#define MEASUREMENT_WIDGET_HPP

#include "MedV4D/Common/Common.h"
#include "ui_MeasurementWidget.h"
#include "MeasurementModule/MeasurementController.hpp"
#include <QtGui>
#include <iostream>
#include <fstream>

class MeasurementWidget: public QWidget, public Ui::MeasurementWidget
{
	Q_OBJECT;
public:
	MeasurementWidget( MeasurementController::Ptr aController ): mController( aController )
	{
		ASSERT( aController.get() != 0 );
		setupUi( this );

		pointsView->setModel( &(aController->getPointModel()) );

		proximalShaftPointsView->setModel( &(aController->getProximalShaftPointModel()) );
	}
public slots:
	void
	startHumeralHeadPointDefining( bool aStart )
	{
		if( aStart ) {
			mController->setMeasurementMode( MeasurementController::mmHUMERAL_HEAD );
		} else {
			mController->setMeasurementMode( MeasurementController::mmNONE );
		}
	}

	void
	startProximalShaftPointDefining( bool aStart )
	{
		if( aStart ) {
			mController->setMeasurementMode( MeasurementController::mmPROXIMAL_SHAFT );
		} else {
			mController->setMeasurementMode( MeasurementController::mmNONE );
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
	MeasurementController::Ptr mController;
};


#endif /*MEASUREMENT_WIDGET_HPP*/

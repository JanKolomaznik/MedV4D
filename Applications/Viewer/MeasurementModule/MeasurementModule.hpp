#ifndef MEASUREMENT_MODULE_HPP
#define MEASUREMENT_MODULE_HPP

#include <QtGui>
#include <QtCore>
#include "MedV4D/GUI/utils/Module.h"
#include "MedV4D/GUI/managers/ApplicationManager.h"
#include "MeasurementModule/MeasurementWidget.hpp"
#include "MeasurementModule/MeasurementController.hpp"
#include "MedV4D/Common/IDGenerator.h"




class MeasurementModule: public AModule
{
public:

	MeasurementModule(): AModule( "Measurement Module" )
	{}

	bool
	isUnloadable();

	void
	startMeasurement();

	void
	stopMeasurement();
protected:
	void
	loadModule();

	void
	unloadModule();

	MeasurementController::Ptr mViewerController;
	M4D::Common::IDNumber mModeId;
};


class StartMeasurementAction: public QAction
{
	Q_OBJECT;
public:
	StartMeasurementAction( MeasurementModule &aModule, QObject *parent )
		: QAction( "Measurement", parent ), mModule( aModule )
	{
		setCheckable( true );
		setChecked( false );
		QObject::connect( this, SIGNAL( toggled(bool) ), this, SLOT( toggleMeasurement(bool) ) );
	}
public slots:
	void
	toggleMeasurement( bool aToggle )
	{
		if( aToggle ) {
			mModule.startMeasurement();
		} else {
			mModule.stopMeasurement();
		}
	}

protected:
	MeasurementModule &mModule;
};

#endif /*MEASUREMENT_MODULE_HPP*/

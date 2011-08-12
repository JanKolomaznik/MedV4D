#ifndef SHOULDER_MEASUREMENT_MODULE_HPP
#define SHOULDER_MEASUREMENT_MODULE_HPP

#include <QtGui>
#include <QtCore>
#include "GUI/utils/Module.h"
#include "GUI/utils/ApplicationManager.h"
#include "ShoulderMeasurementModule/ShoulderMeasurementWidget.hpp"
#include "ShoulderMeasurementModule/ShoulderMeasurementController.hpp"
#include "common/IDGenerator.h"




class ShoulderMeasurementModule: public AModule
{
public:

	void
	load();

	void
	unload();

	bool
	isUnloadable();

	std::string
	getName();

	void
	startMeasurement();

	void
	stopMeasurement();
protected:
	ShoulderMeasurementController::Ptr mViewerController;
	M4D::Common::IDNumber mModeId;
};


class StartMeasurementAction: public QAction
{
	Q_OBJECT;
public:
	StartMeasurementAction( ShoulderMeasurementModule &aModule, QObject *parent )
		: QAction( "Shoulder measurement", parent ), mModule( aModule )
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
	ShoulderMeasurementModule &mModule;
};

#endif /*SHOULDER_MEASUREMENT_MODULE_HPP*/

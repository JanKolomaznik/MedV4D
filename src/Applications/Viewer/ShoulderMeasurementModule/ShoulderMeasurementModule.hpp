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
	load()
	{
		ApplicationManager * appManager = ApplicationManager::getInstance();

		mViewerController = ShoulderMeasurementController::Ptr( new ShoulderMeasurementController );

		M4D::Common::IDNumber modeId = appManager->addNewMode( mViewerController/*controller*/, mViewerController/*renderer*/ );
		mViewerController->setModeId( modeId );
		QObject::connect( mViewerController.get(), SIGNAL( updateRequest() ), appManager, SLOT( updateGUIRequest() ) );
 
		appManager->createDockWidget( "Shoulder Measurement", Qt::RightDockWidgetArea, new ShoulderMeasurementWidget( mViewerController ) );

		//QList<QAction*> &annotationActions = mViewerController->getActions();
		//QToolBar *toolbar = M4D::GUI::createToolbarFromActions( "Annotations toolbar", annotationActions );
		//appManager->addToolBar( toolbar );

		mLoaded = true;
	}

	void
	unload()
	{

	}

	bool
	isUnloadable()
	{
		return false;
	}

	std::string
	getName()
	{
		return "Shoulder Measurement Module";
	}
protected:
	ShoulderMeasurementController::Ptr mViewerController;
};

#endif /*SHOULDER_MEASUREMENT_MODULE_HPP*/

#include "ShoulderMeasurementModule/ShoulderMeasurementModule.hpp"


void
ShoulderMeasurementModule::loadModule()
{
	ApplicationManager * appManager = ApplicationManager::getInstance();

	mViewerController = ShoulderMeasurementController::Ptr( new ShoulderMeasurementController );

	mModeId = appManager->addNewMode( mViewerController/*controller*/, mViewerController/*renderer*/ );
	mViewerController->setModeId( mModeId );
	QObject::connect( mViewerController.get(), SIGNAL( updateRequest() ), appManager, SLOT( updateGUIRequest() ) );


	//QList<QAction*> &annotationActions = mViewerController->getActions();
	QToolBar *toolbar = new QToolBar( "Shoulder measurement toolbar" );
	toolbar->addAction( new StartMeasurementAction( *this, NULL ) );
	appManager->addToolBar( toolbar );

	mLoaded = true;
}

void
ShoulderMeasurementModule::unloadModule()
{

}

bool
ShoulderMeasurementModule::isUnloadable()
{
	return false;
}

void
ShoulderMeasurementModule::startMeasurement()
{
	ApplicationManager * appManager = ApplicationManager::getInstance();
	appManager->createDockWidget( "Shoulder Measurement", Qt::RightDockWidgetArea, new ShoulderMeasurementWidget( mViewerController ) );

	appManager->activateMode( mModeId );
}

void
ShoulderMeasurementModule::stopMeasurement()
{

}


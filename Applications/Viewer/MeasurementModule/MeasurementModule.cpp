#include "MeasurementModule/MeasurementModule.hpp"


void
MeasurementModule::loadModule()
{
	ApplicationManager * appManager = ApplicationManager::getInstance();

	mViewerController = MeasurementController::Ptr( new MeasurementController );

	mModeId = appManager->addNewMode( mViewerController/*controller*/, mViewerController/*renderer*/ );
	mViewerController->setModeId( mModeId );
	QObject::connect( mViewerController.get(), SIGNAL( updateRequest() ), appManager, SLOT( updateGUIRequest() ) );


	//QList<QAction*> &annotationActions = mViewerController->getActions();
	QToolBar *toolbar = new QToolBar( "Measurement Toolbar" );
	toolbar->addAction( new StartMeasurementAction( *this, NULL ) );
	appManager->addToolBar( toolbar );

	mLoaded = true;
}

void
MeasurementModule::unloadModule()
{

}

bool
MeasurementModule::isUnloadable()
{
	return false;
}

void
MeasurementModule::startMeasurement()
{
	ApplicationManager * appManager = ApplicationManager::getInstance();
	appManager->createDockWidget( "Measurement Tool", Qt::RightDockWidgetArea, new MeasurementWidget( mViewerController ) );

	ASSERT( mModeId > 0 );
	appManager->activateMode( mModeId );
}

void
MeasurementModule::stopMeasurement()
{

}


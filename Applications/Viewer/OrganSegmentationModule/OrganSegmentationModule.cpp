#include "OrganSegmentationModule/OrganSegmentationModule.hpp"
#include "OrganSegmentationModule/OrganSegmentationController.hpp"
#include "OrganSegmentationModule/OrganSegmentationWidget.hpp"
void
OrganSegmentationModule::loadModule()
{
	ApplicationManager * appManager = ApplicationManager::getInstance();

	mViewerController = OrganSegmentationController::Ptr( new OrganSegmentationController );

	mModeId = appManager->addNewMode( mViewerController/*controller*/, mViewerController/*renderer*/ );
	mViewerController->setModeId( mModeId );
	QObject::connect( mViewerController.get(), SIGNAL( updateRequest() ), appManager, SLOT( updateGUIRequest() ) );


	//QList<QAction*> &annotationActions = mViewerController->getActions();
	QToolBar *toolbar = new QToolBar( "Organ segmentation toolbar" );
	toolbar->addAction( new StartOrganSegmentationAction( *this, NULL ) );
	appManager->addToolBar( toolbar );

	mLoaded = true;
}

void
OrganSegmentationModule::unloadModule()
{

}

bool
OrganSegmentationModule::isUnloadable()
{
	return false;
}

void
OrganSegmentationModule::startSegmentationMode()
{
	ApplicationManager * appManager = ApplicationManager::getInstance();
	appManager->createDockWidget( "Organ segmentation", Qt::RightDockWidgetArea, new OrganSegmentationWidget( mViewerController ) );

	appManager->activateMode( mModeId );
}

void
OrganSegmentationModule::stopSegmentationMode()
{
	
}

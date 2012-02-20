#include "OrganSegmentationModule/OrganSegmentationModule.hpp"
#include "OrganSegmentationModule/OrganSegmentationController.hpp"
#include "OrganSegmentationModule/OrganSegmentationWidget.hpp"

#include "MedV4D/GUI/managers/DatasetManager.h"
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
OrganSegmentationModule::createMask()
{
	ImageRecord::Ptr imageRecord = DatasetManager::getInstance()->getCurrentImageInfo();
	if( imageRecord && imageRecord->image ) {
		const M4D::Imaging::AImageDim<3> & image = M4D::Imaging::AImageDim<3>::Cast( *(imageRecord->image) );
		mMask = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< typename M4D::Imaging::Mask3D::Element, 3 >( image.GetImageExtentsRecord() );
		
		DatasetManager::getInstance()->secondaryImageInputConnection().PutDataset( mMask );
		mViewerController->mMask = mMask;
	}
}

void
OrganSegmentationModule::startSegmentationMode()
{
	ApplicationManager * appManager = ApplicationManager::getInstance();
	appManager->createDockWidget( "Organ segmentation", Qt::RightDockWidgetArea, new OrganSegmentationWidget( mViewerController, *this ) );

	appManager->activateMode( mModeId );
}

void
OrganSegmentationModule::stopSegmentationMode()
{
	
}

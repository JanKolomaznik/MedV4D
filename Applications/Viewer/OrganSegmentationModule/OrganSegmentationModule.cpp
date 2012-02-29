#include "OrganSegmentationModule/OrganSegmentationModule.hpp"
#include "OrganSegmentationModule/OrganSegmentationController.hpp"
#include "OrganSegmentationModule/OrganSegmentationWidget.hpp"

#include "MedV4D/GUI/managers/DatasetManager.h"

#include <QtGui>
#include <QtCore>

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
OrganSegmentationModule::loadMask()
{
	try {
	QString fileName = QFileDialog::getOpenFileName(/*ApplicationManager::getInstance()->*/NULL, /*tr(*/"Load Mask"/*)*/ );

	if ( !fileName.isEmpty() ) {

		DatasetID mDatasetId = DatasetManager::getInstance()->openFileBlocking( std::string( fileName.toLocal8Bit().data() ) );
		
		ADatasetRecord::Ptr rec = DatasetManager::getInstance()->getDatasetInfo( mDatasetId );
		if ( !rec ) {
			D_PRINT( "Loaded dataset record not available" );
			return;
		}
		ImageRecord * iRec = dynamic_cast< ImageRecord * >( rec.get() );
		if ( !iRec ) {
			D_PRINT( "Loaded dataset isn't image" );
		}
		M4D::Imaging::AImage::Ptr image = iRec->image;
		mMask = M4D::Imaging::Mask3D::Cast( image );
		
		DatasetManager::getInstance()->secondaryImageInputConnection().PutDataset( mMask );
		mViewerController->mMask = mMask;
	}
	} catch ( std::exception &e ) {
		QMessageBox::critical ( NULL, "Exception", QString( e.what() ) );
	}
	catch (...) {
		QMessageBox::critical ( NULL, "Exception", "Problem with file loading" );
	}
}

void 
OrganSegmentationModule::updateTimestamp()
{
	ASSERT( mMask );
	
	M4D::Imaging::WriterBBoxInterface & mod = mMask->SetWholeDirtyBBox();
	mod.SetModified();
	
	DatasetManager::getInstance()->secondaryImageInputConnection().PutDataset( mMask );
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

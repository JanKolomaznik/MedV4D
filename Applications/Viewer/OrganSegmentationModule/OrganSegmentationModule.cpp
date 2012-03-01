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
//************************************************************************************
M4D::Imaging::Mask3D::SliceRegion::PointType 
FindMaskCenterOfGravity( const M4D::Imaging::Mask3D::SliceRegion &region )
{
	MaskType::SliceRegion::PointType sum;
	MaskType::SliceRegion::PointType min = region.GetMinimum();
	MaskType::SliceRegion::PointType idx;
	MaskType::SliceRegion::PointType max = region.GetMaximum();
	int32 count = 0;
	for( idx = min; idx[1] < max[1]; ++idx[1] ) {
		for( idx[0] = min[0]; idx[0] < max[0]; ++idx[0] ) {
			//LOG( idx << " -> " << (int16)region.GetElement( idx ) );
			if( region.GetElement( idx ) != 0 ) {
				++count;
				sum += idx;
			}
		}
	}
	
	if( count == 0 ) {
		_THROW_ M4D::ErrorHandling::ExceptionBase( "Center of gravity unable to find." );
	}
	return M4D::Imaging::Mask3D::SliceRegion::PointType( sum[0] / count, sum[1] / count );
}

void
GetPoles( const M4D::Imaging::Mask3D & mask, M4D::Imaging::Mask3D::PointType &north, M4D::Imaging::Mask3D::PointType &south )
{
	int32 southSliceCoord = mask.GetMinimum()[2];
	int32 northSliceCoord = mask.GetMaximum()[2]-1;
	M4D::Imaging::Mask3D::SliceRegion southRegion = mask.GetSlice( southSliceCoord );
	M4D::Imaging::Mask3D::SliceRegion northRegion = mask.GetSlice( northSliceCoord );

	/*M4D::Imaging::Mask2D::Ptr tmp = mask.GetRestrictedImage( southRegion );
	ImageFactory::DumpImage( "pom.dump", *tmp );
	tmp = mask.GetRestrictedImage( northRegion );
	ImageFactory::DumpImage( "pom2.dump", *tmp );*/

	MaskType::SliceRegion::PointType southTmp = FindMaskCenterOfGravity( southRegion );
	MaskType::SliceRegion::PointType northTmp = FindMaskCenterOfGravity( northRegion );

	south = M4D::Imaging::Mask3D::PointType( southTmp[0], southTmp[1], southSliceCoord );
	north = M4D::Imaging::Mask3D::PointType( northTmp[0], northTmp[1], northSliceCoord );

	D_PRINT( "South pole = " << south );
	D_PRINT( "North pole = " << north );
}

void
OrganSegmentationModule::computeStats()
{
	M4D::Imaging::Mask3D::PointType north;
	M4D::Imaging::Mask3D::PointType south;
	GetPoles( *mMask, north, south );
	M4D::Imaging::Transformation trans = M4D::Imaging::GetTransformation ( north, south, mMask->GetElementExtents() );
}

void
OrganSegmentationModule::loadModel()
{
	try {
	QString fileName = QFileDialog::getOpenFileName(/*ApplicationManager::getInstance()->*/NULL, /*tr(*/"Load Mask"/*)*/ );

	if ( !fileName.isEmpty() ) {
		mProbModel = M4D::Imaging::CanonicalProbModel::LoadFromFile( std::string( fileName.toLocal8Bit().data() ) );
	}
	} catch ( std::exception &e ) {
		QMessageBox::critical ( NULL, "Exception", QString( e.what() ) );
	}
	catch (...) {
		QMessageBox::critical ( NULL, "Exception", "Problem with model loading" );
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

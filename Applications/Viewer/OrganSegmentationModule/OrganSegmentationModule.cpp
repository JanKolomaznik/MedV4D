#include "OrganSegmentationModule/OrganSegmentationModule.hpp"
#include "OrganSegmentationModule/OrganSegmentationController.hpp"
#include "OrganSegmentationModule/OrganSegmentationWidget.hpp"

#include "MedV4D/GUI/managers/DatasetManager.h"

#include "MedV4D/Imaging/cuda/ConnectedComponentLabeling.h"
#include "MedV4D/Imaging/cuda/EdgeDetection.h"
#include "MedV4D/Imaging/cuda/LocalMinimaDetection.h"
#include "MedV4D/Imaging/cuda/WatershedTransformation.h"
#include "MedV4D/Imaging/cuda/GraphOperations.h"
#include "MedV4D/Imaging/cuda/SimpleFilters.h"
#include "MedV4D/Imaging/ImageFactory.h"

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
		M4D::Imaging::Mask3D::Ptr tmpMask = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< typename M4D::Imaging::Mask3D::Element, 3 >( image.GetImageExtentsRecord() );
		
		prepareMask( tmpMask );
	}
}

void
OrganSegmentationModule::prepareMask( M4D::Imaging::Mask3D::Ptr aMask )
{
	mMask = aMask;
	DatasetManager::getInstance()->secondaryImageInputConnection().PutDataset( mMask );
	mViewerController->mMask = mMask;
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
		M4D::Imaging::Mask3D::Ptr tmpMask = M4D::Imaging::Mask3D::Cast( image );
		
		prepareMask( tmpMask );
	}
	} catch ( std::exception &e ) {
		QMessageBox::critical ( NULL, "Exception", QString( e.what() ) );
	}
	catch (...) {
		QMessageBox::critical ( NULL, "Exception", "Problem with file loading" );
	}
}

void 
OrganSegmentationModule::loadIndexFile()
{
	try {
	QString fileName = QFileDialog::getOpenFileName(/*ApplicationManager::getInstance()->*/NULL, /*tr(*/"Load Mask"/*)*/ );

	if ( !fileName.isEmpty() ) {
		std::fstream file( fileName.toLocal8Bit().data() );
		std::string first, second;
		
		file >> first;
		file >> second;
		file.close();
		
		Path dir( fileName.toLocal8Bit().data() );
		dir = dir.parent_path();

		
		Path imageFile = dir;
		imageFile /= first;
		Path maskFile = dir;
		maskFile /= second;
		
			
			//D_PRINT( "Loading training image number '" << i << "' from file '" << imageFile.string() <<"'." );
			M4D::Imaging::AImage::Ptr aimage = M4D::Imaging::ImageFactory::LoadDumpedImage( imageFile.string() );
			mImage = Image16_3D::Cast( aimage );
			DatasetManager::getInstance()->primaryImageInputConnection().PutDataset( aimage );
			
			//D_PRINT( "Loading training mask number '" << i << "' from file '" << maskFile.string() <<"'." );
			aimage = M4D::Imaging::ImageFactory::LoadDumpedImage( maskFile.string() );
			M4D::Imaging::Mask3D::Ptr mask = M4D::Imaging::Mask3D::Cast( aimage );
			
			prepareMask(mask);
			
		
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
	M4D::Imaging::Mask3D::SliceRegion::PointType sum;
	M4D::Imaging::Mask3D::SliceRegion::PointType min = region.GetMinimum();
	M4D::Imaging::Mask3D::SliceRegion::PointType idx;
	M4D::Imaging::Mask3D::SliceRegion::PointType max = region.GetMaximum();
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

	M4D::Imaging::Mask3D::SliceRegion::PointType southTmp = FindMaskCenterOfGravity( southRegion );
	M4D::Imaging::Mask3D::SliceRegion::PointType northTmp = FindMaskCenterOfGravity( northRegion );

	south = M4D::Imaging::Mask3D::PointType( southTmp[0], southTmp[1], southSliceCoord );
	north = M4D::Imaging::Mask3D::PointType( northTmp[0], northTmp[1], northSliceCoord );

	D_PRINT( "South pole = " << south );
	D_PRINT( "North pole = " << north );
}

void
OrganSegmentationModule::computeStats()
{
	if( !mImage || !mMask || !mProbModel ) {
		QMessageBox::critical ( NULL, "Exception", QString( "Neco chybi" ) );
		return;
	}
	M4D::Imaging::Mask3D::PointType north;
	M4D::Imaging::Mask3D::PointType south;
	M4D::Imaging::Mask3D::PointType idx;
	M4D::Imaging::Mask3D::PointType minimum = mImage->GetMinimum();
	M4D::Imaging::Mask3D::PointType maximum = mImage->GetMaximum();
	Vector3f rmin = mImage->GetRealMinimum();
	Vector3f rmax = mImage->GetRealMaximum();
	Vector3f extents = mMask->GetElementExtents();
	GetPoles( *mMask, north, south );
	M4D::Imaging::Transformation trans = M4D::Imaging::GetTransformation ( north, south, extents );
	
	int count = 0;
	int failOut = 0;
	int failIn = 0;
	int ok = 0;
	int maskCount = 0;
	for( idx[2] = minimum[2]; idx[2] < maximum[2]; ++idx[2] ) {
		for( idx[1] = minimum[1]; idx[1] < maximum[1]; ++idx[1] ) {
			for( idx[0] = minimum[0]; idx[0] < maximum[0]; ++idx[0] ) {
				int16 val = mImage->GetElement( idx );
				uint8 maskVal = mMask->GetElement( idx );
				Vector3f pos = rmin + VectorMemberProduct<int,float,3>(idx-minimum, extents );
				//float ratio = mProbModel->LogRatioProbabilityIntesityPositionDependent( val, trans( pos ) );
				float ratio = mProbModel->LogRatioCombination( trans( pos ), val, 0.0f, 0.0f, 1.0f );
				float threshold = 0.3;
				if( ratio < threshold && maskVal > 100 ) ++failOut;
				if( ratio >= threshold && maskVal < 100 ) ++failIn;
				if( ratio >= threshold && maskVal > 100 ) ++ok;
				if( maskVal > 100 ) ++maskCount;
				//D_PRINT( ratio );
				++count;
			}	
		}	
	}
	D_PRINT( "count = " << count );
	D_PRINT( "maskCount = " << maskCount );
	D_PRINT( "failIn = " << failIn << " : " << float( failIn )/ maskCount * 100.0f );
	D_PRINT( "failOut = " << failOut << " : " << float( failOut )/ maskCount  * 100.0f );
	D_PRINT( "ok = " << ok << " : " << float( ok )/ maskCount  * 100.0f );
	D_PRINT( "size = " << mImage->GetSize() << "; " << VectorCoordinateProduct(mImage->GetSize()) );
	
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

void
OrganSegmentationModule::computeWatershedTransformation()
{
	ImageRecord::Ptr imageRecord = DatasetManager::getInstance()->getCurrentImageInfo();
	if( imageRecord && imageRecord->image ) {
		
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( imageRecord->image->GetElementTypeID(),
			TTYPE threshold = TypeTraits<TTYPE>::Max;
			
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::Ptr typedImage = IMAGE_TYPE::Cast( imageRecord->image );
			IMAGE_TYPE::Ptr gradientImage = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< TTYPE, 3 >( typedImage->GetMinimum(), typedImage->GetMaximum(), typedImage->GetElementExtents() );

			Sobel3D( typedImage->GetRegion(), gradientImage->GetRegion(), static_cast< TTYPE >( 0 ) );
			mGradientImage = gradientImage;
			
			
			mWatersheds = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< uint32, 3 >( typedImage->GetMinimum(), typedImage->GetMaximum(), typedImage->GetElementExtents() );
			
			std::cout << "Finding local minima ..."; std::cout.flush();
			LocalMinimaRegions3D( gradientImage->GetRegion(), mWatersheds->GetRegion(), threshold );
			std::cout << "Done\n";

			std::cout << "Watershed transformation ..."; std::cout.flush();
			WatershedTransformation3D( mWatersheds->GetRegion(), typedImage->GetRegion(), mWatersheds->GetRegion() );
			std::cout << "Done\n";
		);
		//DatasetManager::getInstance()->secondaryImageInputConnection().PutDataset( mWatersheds );
		//mViewerController->mMask = mMask;
	}
}

void
OrganSegmentationModule::computeSegmentation()
{
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( mGradientImage->GetElementTypeID(),
		typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
		IMAGE_TYPE::Ptr typedGradientImage = IMAGE_TYPE::Cast( mGradientImage );
		
		pushRelabelMaxFlow( mWatersheds->GetRegion(), typedGradientImage->GetRegion() );
		
	);
}

/*void
OrganSegmentationModule::bimaskFromSelectedWatersheds()
{

}*/

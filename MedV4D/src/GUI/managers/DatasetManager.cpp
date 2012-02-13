#include "MedV4D/GUI/managers/DatasetManager.h"
#include "MedV4D/DICOMInterface/DcmProvider.h"

DatasetManager *fileManagerInstance = NULL;

DatasetManager *
DatasetManager::getInstance()
{
	ASSERT( fileManagerInstance && fileManagerInstance->mInitialized );
	return fileManagerInstance;
}

DatasetManager::DatasetManager()
	: mInitialized( false )
{
	ASSERT( fileManagerInstance == NULL )

	fileManagerInstance = this;
}

void
DatasetManager::initialize()
{
	ASSERT( fileManagerInstance != NULL )
	mInitialized = true;
}

void
DatasetManager::finalize()
{
	ASSERT( mInitialized );
}

DatasetManager::~DatasetManager()
{
	finalize();
}

ADatasetRecord::Ptr
DatasetManager::getDatasetInfo( DatasetID aDatasetId )
{
	M4D::Multithreading::ScopedLock lock( mDatasetInfoAccessLock );

	DatasetInfoMap::iterator it = mDatasetInfos.find( aDatasetId );
	if ( it != mDatasetInfos.end() ) {
		return it->second;
	}
	return ADatasetRecord::Ptr();
}

DatasetID
DatasetManager::openFileNonBlocking( boost::filesystem::path aPath, ProgressNotifier::Ptr aProgressNotifier )
{
	if (!aProgressNotifier) {
		_THROW_ EBadParameter( "Invalid progress notifier passed as parameter!" );
	}
	DatasetID id = mIdGenerator.NewID();
	
	boost::thread th = boost::thread( 
			boost::bind(
			&DatasetManager::openFileHelper,  
			this,
			aPath, 
			aProgressNotifier,
			id
			)
			);
	th.detach();

	return id;
}

DatasetID
DatasetManager::openFileBlocking( boost::filesystem::path aPath, ProgressNotifier::Ptr aProgressNotifier )
{
	DatasetID id = mIdGenerator.NewID();
	openFileHelper( aPath, aProgressNotifier, id );
	return id;
}


void
DatasetManager::openFileHelper( boost::filesystem::path aPath, ProgressNotifier::Ptr aProgressNotifier, DatasetID aDatasetId )
{
	M4D::Imaging::AImage::Ptr image;
	if ( aPath.extension() == ".dcm" || aPath.extension() == ".DCM" ) {
		M4D::Dicom::DicomObjSetPtr dicomObjSet = M4D::Dicom::DicomObjSetPtr( new M4D::Dicom::DicomObjSet() );
		M4D::Dicom::DcmProvider::LoadSerieThatFileBelongsTo( aPath, aPath.parent_path(), *dicomObjSet, aProgressNotifier );
		image = M4D::Dicom::DcmProvider::CreateImageFromDICOM( dicomObjSet );
	} else {
		image = M4D::Imaging::ImageFactory::LoadDumpedImage( aPath.string() );
	}

	if (image) {
		registerImage( aDatasetId, aPath, image );
	}
	aProgressNotifier->finished();
	//mProdconn.PutDataset( image );
}

void
DatasetManager::registerImage( DatasetID aDatasetId, boost::filesystem::path aPath, M4D::Imaging::AImage::Ptr aImage )
{
	ASSERT( aDatasetId != 0 );
	M4D::Multithreading::ScopedLock lock( mDatasetInfoAccessLock );

	ImageRecord *rec = new ImageRecord();
	ADatasetRecord::Ptr p = ADatasetRecord::Ptr( rec );

	rec->id = aDatasetId;
	rec->filePath = aPath;
	rec->image = aImage;
	mDatasetInfos[ aDatasetId ] = p;
	D_PRINT( "Dataset info added, id = " << aDatasetId << ", dataset count = " << mDatasetInfos.size() );
}


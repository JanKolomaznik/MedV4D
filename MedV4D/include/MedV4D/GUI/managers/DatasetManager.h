#ifndef DATASET_MANAGER_H
#define DATASET_MANAGER_H

#include <QtCore>
#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/Thread.h"
#include "MedV4D/Common/ProgressNotifier.h"
#include "MedV4D/Imaging/ImageFactory.h"
#include "MedV4D/Imaging/ConnectionInterface.h"

#include <boost/thread/recursive_mutex.hpp>

typedef M4D::Common::IDNumber DatasetID;
struct ADatasetRecord
{
	typedef std::shared_ptr< ADatasetRecord > Ptr;
	
	DatasetID id;

	virtual ~ADatasetRecord()
	{}
};


struct ImageRecord: public ADatasetRecord 
{
	typedef std::shared_ptr< ImageRecord > Ptr;
	
	boost::filesystem::path	filePath;

	M4D::Imaging::AImage::Ptr image;
};

class DatasetManager
{
public:
	static DatasetManager *
	getInstance();

	DatasetManager();

	virtual void
	initialize();

	virtual void
	finalize();

	~DatasetManager();

	virtual DatasetID
	openFileNonBlocking( boost::filesystem::path aPath, ProgressNotifier::Ptr aProgressNotifier, bool aUseAsCurrent = true );

	virtual DatasetID
	openFileBlocking( boost::filesystem::path aPath, ProgressNotifier::Ptr aProgressNotifier = ProgressNotifier::Ptr(), bool aUseAsCurrent = true );

	ADatasetRecord::Ptr
	getDatasetInfo( DatasetID aDatasetId );
	
	ImageRecord::Ptr
	getImageInfo( DatasetID aDatasetId );
	
	ADatasetRecord::Ptr
	getCurrentDatasetInfo() 
	{
		return getDatasetInfo( mCurrentDatasetId );
	}
	
	ImageRecord::Ptr
	getCurrentImageInfo() 
	{
		return getImageInfo( mCurrentDatasetId );
	}
	
	void
	setCurrentDatasetInfo( DatasetID aDatasetId );
	
	M4D::Imaging::ConnectionInterface &
	primaryImageInputConnection()
	{
		return mPrimaryProdconn;
	}
	
	M4D::Imaging::ConnectionInterface &
	secondaryImageInputConnection()
	{
		return mSecondaryProdconn;
	}
	
	void
	clearAll();

protected:
	typedef std::map< DatasetID, ADatasetRecord::Ptr > DatasetInfoMap;

	virtual void
	openFileHelper( boost::filesystem::path aPath, ProgressNotifier::Ptr aProgressNotifier, DatasetID aDatasetId, bool aUseAsCurrent );

	void
	registerImage( DatasetID aDatasetId, boost::filesystem::path aPath, M4D::Imaging::AImage::Ptr aImage, bool aUseAsCurrent );

	bool mInitialized;
	M4D::Common::IDGenerator mIdGenerator;

	DatasetInfoMap mDatasetInfos;
	//M4D::Multithreading::RecursiveMutex mDatasetInfoAccessLock;
	
	boost::recursive_mutex mDatasetInfoAccessLock;
	
	DatasetID mCurrentDatasetId;
	
	M4D::Imaging::ConnectionTyped< M4D::Imaging::AImage > mPrimaryProdconn;
	M4D::Imaging::ConnectionTyped< M4D::Imaging::AImage > mSecondaryProdconn;

};

#endif /*DATASET_MANAGER_H*/

#ifndef DATASET_MANAGER_H
#define DATASET_MANAGER_H

#include <QtCore>
#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/Thread.h"
#include "MedV4D/Common/ProgressNotifier.h"
#include "Imaging/ImageFactory.h"

typedef M4D::Common::IDNumber DatasetID;
struct ADatasetRecord
{
	typedef boost::shared_ptr< ADatasetRecord > Ptr;
	
	DatasetID id;

	virtual ~ADatasetRecord()
	{}
};


struct ImageRecord: public ADatasetRecord 
{
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
	openFileNonBlocking( boost::filesystem::path aPath, ProgressNotifier::Ptr aProgressNotifier );

	virtual DatasetID
	openFileBlocking( boost::filesystem::path aPath, ProgressNotifier::Ptr aProgressNotifier = ProgressNotifier::Ptr() );

	ADatasetRecord::Ptr
	getDatasetInfo( DatasetID aDatasetId );

protected:
	typedef std::map< DatasetID, ADatasetRecord::Ptr > DatasetInfoMap;

	virtual void
	openFileHelper( boost::filesystem::path aPath, ProgressNotifier::Ptr aProgressNotifier, DatasetID aDatasetId );

	void
	registerImage( DatasetID aDatasetId, boost::filesystem::path aPath, M4D::Imaging::AImage::Ptr aImage );

	bool mInitialized;
	M4D::Common::IDGenerator mIdGenerator;

	DatasetInfoMap mDatasetInfos;
	M4D::Multithreading::Mutex mDatasetInfoAccessLock;

};

#endif /*DATASET_MANAGER_H*/

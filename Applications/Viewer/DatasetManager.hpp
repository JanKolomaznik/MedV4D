//#pragma once

#ifndef VIEWER_DATASET_MANAGER_HPP
#define VIEWER_DATASET_MANAGER_HPP

#include <unordered_map>
#include <vector>
#include <limits>
#include <atomic>
#include <future>

#include <QFileDialog>

#include <boost/filesystem.hpp>
#include <boost/variant.hpp>

#include <prognot/prognot.hpp>

#include "MedV4D/Imaging/Imaging.h"

class ImageRecord {
public:
	void
	assignImage(M4D::Imaging::AImage::Ptr aImage) {
		mImage = aImage;
	}

//protected:
	//std::atomic_bool mIsReady;
	M4D::Imaging::AImage::Ptr mImage;
};

class DatasetManager {
public:
	DatasetManager()
		: mLastId(0)
	{}

	typedef int64_t DatasetID;

	DatasetID
	loadFromFile();

	ImageRecord &
	getDatasetRecord(DatasetID aId)
	{
		return mImages[aId];
	}

	std::future<M4D::Imaging::AImage::Ptr>
	openFileNonBlocking(boost::filesystem::path aPath, prognot::ProgressNotifier aProgressNotifier);

	M4D::Imaging::AImage::Ptr
	openFileBlocking(boost::filesystem::path aPath, prognot::ProgressNotifier aProgressNotifier);

protected:

	DatasetID
	newID()
	{
		return ++mLastId;
	}

	std::unordered_map<DatasetID, ImageRecord> mImages;
	DatasetID mLastId;

	QFileDialog mFileDialog;
};


#endif //VIEWER_DATASET_MANAGER_HPP

//#pragma once

#ifndef VIEWER_DATASET_MANAGER_HPP
#define VIEWER_DATASET_MANAGER_HPP

#include <unordered_map>
#include <vector>
#include <limits>
#include <atomic>
#include <future>
#include <algorithm>

#include <QFileDialog>

#include <boost/filesystem.hpp>
#include <boost/variant.hpp>

#include <prognot/prognot.hpp>

#include <QAbstractListModel>

#include "MedV4D/Imaging/Imaging.h"
#include "MedV4D/GUI/widgets/GeneralViewer.h"

class DatasetManager;

class ImageListModel : public QAbstractListModel
{
public:
	friend class DatasetManager;

	ImageListModel(DatasetManager &aManager)
		: QAbstractListModel(nullptr)
		, mManager(aManager)
	{}

	int
	rowCount(const QModelIndex & parent = QModelIndex()) const override;
	int
	columnCount(const QModelIndex & parent = QModelIndex()) const override
	{
		return 1;
	}
	QVariant
	data(const QModelIndex & index, int role = Qt::DisplayRole) const override;
protected:

	DatasetManager &mManager;
};


class ImageRecord {
public:
	void
	assignImage(M4D::Imaging::AImage::Ptr aImage, boost::filesystem::path aPath) {
		mImage = aImage;
		mFileName = aPath;
	}

	std::string
	name() const
	{
		return mFileName.filename().string();
	}

//protected:
	//std::atomic_bool mIsReady;
	M4D::Imaging::AImage::Ptr mImage;
	boost::filesystem::path mFileName;
};



class DatasetManager {
public:
	friend class ImageListModel;

	DatasetManager()
		: mLastId(0)
		, mImageModel(*this)
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

	ImageListModel &
	imageModel()
	{
		return mImageModel;
	}

	int
	indexFromID(DatasetID aId) const
	{
		auto it = std::find(begin(mDatasetIDList), end(mDatasetIDList), aId);

		if (it != end(mDatasetIDList)) {
			return it - begin(mDatasetIDList);
		}
		return -1;
	}
	DatasetID
	idFromIndex(int aIndex) const
	{
		return mDatasetIDList.at(aIndex);
	}
protected:
	DatasetID
	newID()
	{
		mImageModel.beginInsertRows(QModelIndex(), mDatasetIDList.size(), mDatasetIDList.size());
		mDatasetIDList.push_back(++mLastId); //TODO what to do on exception
		mImageModel.endInsertRows();
		return mLastId;
	}

	/*ImageRecord &
	findOrCreateRecord(DatasetID aId)
	{
		if (!mImages.count(aId)) {
			mDatasetIDList.push_back(aId);
		}
		return mImages[aId];
	}*/

	std::unordered_map<DatasetID, ImageRecord> mImages;
	DatasetID mLastId;

	QFileDialog mFileDialog;

	std::vector<DatasetManager::DatasetID> mDatasetIDList;

	ImageListModel mImageModel;
};

class ViewerInputDataWithId : public M4D::GUI::Viewer::ViewerInputData
{
public:
	typedef std::shared_ptr<ViewerInputDataWithId> Ptr;
	typedef std::shared_ptr<const ViewerInputDataWithId> ConstPtr;

	ViewerInputDataWithId()
		: mPrimaryImageId(-1)
		, mSecondaryImageId(-1)
	{}

	ViewerInputDataWithId(
		M4D::Imaging::AImageDim<3>::ConstPtr aPrimaryImage,
		DatasetManager::DatasetID aPrimaryImageId,
		M4D::Imaging::AImageDim<3>::ConstPtr aSecondaryImage = M4D::Imaging::AImageDim<3>::ConstPtr(),
		DatasetManager::DatasetID aSecondaryImageId = -1
		)
		: M4D::GUI::Viewer::ViewerInputData(aPrimaryImage, aSecondaryImage)
		, mPrimaryImageId(aPrimaryImageId)
		, mSecondaryImageId(aSecondaryImageId)
	{}

	void
	assignPrimaryImageWithId(
		M4D::Imaging::AImageDim<3>::ConstPtr aImage,
		DatasetManager::DatasetID aImageId
		)
	{
		mPrimaryImage = aImage;
		mPrimaryImageId = aImageId;
	}

	void
	assignSecondaryImageWithId(
		M4D::Imaging::AImageDim<3>::ConstPtr aImage,
		DatasetManager::DatasetID aImageId
		)
	{
		mSecondaryImage = aImage;
		mSecondaryImageId = aImageId;
	}

	DatasetManager::DatasetID
	primaryImageId() const
	{
		return mPrimaryImageId;
	}

	DatasetManager::DatasetID
	secondaryImageId() const
	{
		return mSecondaryImageId;
	}
protected:
	DatasetManager::DatasetID mPrimaryImageId;
	DatasetManager::DatasetID mSecondaryImageId;
};


#endif //VIEWER_DATASET_MANAGER_HPP

#include "DatasetManager.hpp"

#include <QApplication>
#include <QMessageBox>
#include <future>
#include <memory>

#include "MedV4D/Imaging/ImageFactory.h"
#include "MedV4D/DICOMInterface/DcmProvider.h"

#include <prognot/Qt/ProgressDialog.hpp>

#include "MedV4D/Common/MathTools.h"
#include "MedV4D/Imaging/ImageFactory.h"
#include "MedV4D/Imaging/AImage.h"
#include "MedV4D/Imaging/Image.h"

#include "ItkUtils.hpp"


M4D::Imaging::AImage::Ptr
loadDataFile(boost::filesystem::path aPath, prognot::ProgressNotifier aProgressNotifier)
{
	FILE *fp = fopen(aPath.string().c_str(), "rb");

	unsigned short vuSize[3];
	fread((void*)vuSize, 3, sizeof(unsigned short), fp);

	auto image = M4D::Imaging::ImageFactory::CreateEmptyImage3DTyped
			<unsigned short>(vuSize[0], vuSize[1], vuSize[2]);


	int uCount = int(vuSize[0])*int(vuSize[1])*int(vuSize[2]);
	//unsigned short *pData = new unsigned short[uCount];
	fread((void*)image->GetPointer(), uCount, sizeof(unsigned short), fp);

	fclose(fp);
	return image;
}

DatasetManager::DatasetManager()
	: mLastId(0)
	, mImageModel(*this)
	, mFileDialog(QApplication::activeWindow())
{

}

DatasetID
DatasetManager::loadFromFile()
{
	QStringList fileNames;
	QString fileName;
	if ( mFileDialog.exec() ) {
		fileNames = mFileDialog.selectedFiles();
		if (!fileNames.isEmpty()) {
			fileName = fileNames[0];
		}
	}
	if (fileName.isEmpty()) {
		return 0;
	}
	try {
		prognot::qt::ProgressDialog progressDialog(QApplication::activeWindow());
		//TODO QString to path proper conversion on win
		auto image = openFileNonBlocking(fileName.toStdString(), progressDialog.progressNotifier());
		progressDialog.exec();

		ImageRecord currentRecord;
		currentRecord.assignImage(image.get(), boost::filesystem::path(fileName.toStdString()));
		DatasetID id = addNewRecord(std::move(currentRecord));
		emit registeredNewDataset(id); // TODO - single place where to emit this
		return id;
	} catch ( std::exception &e ) {
		QMessageBox::critical ( NULL, "Exception", QString( e.what() ) );
		return 0;
	}
}

std::future<M4D::Imaging::AImage::Ptr>
DatasetManager::openFileNonBlocking(boost::filesystem::path aPath, prognot::ProgressNotifier aProgressNotifier)
{
	// TODO when compilers fix passing movable types to async
	std::shared_ptr<prognot::ProgressNotifier> notifierPtr(new prognot::ProgressNotifier(std::move(aProgressNotifier)));
	return std::async(
		std::launch::async,
		[this, aPath, notifierPtr] () {
			return this->openFileBlocking(aPath, std::move(*notifierPtr));
		});
}

M4D::Imaging::AImage::Ptr
DatasetManager::openFileBlocking(boost::filesystem::path aPath, prognot::ProgressNotifier aProgressNotifier)
{
	aProgressNotifier.setStepCount(3);
	M4D::Imaging::AImage::Ptr image;
	if ( aPath.extension() == ".dcm" || aPath.extension() == ".DCM" ) {
		M4D::Dicom::DicomObjSetPtr dicomObjSet = M4D::Dicom::DicomObjSetPtr( new M4D::Dicom::DicomObjSet() );
		M4D::Dicom::DcmProvider::LoadSerieThatFileBelongsTo( aPath, aPath.parent_path(), *dicomObjSet, aProgressNotifier.subTaskNotifier(1));
		image = M4D::Dicom::DcmProvider::CreateImageFromDICOM( dicomObjSet );
	} else {
		if ( aPath.extension() == ".dat" || aPath.extension() == ".DAT" ) { //TODO better extension checking
			image = loadDataFile(aPath, aProgressNotifier.subTaskNotifier(1));
		} else {
			try {
				image = M4D::Imaging::ImageFactory::LoadDumpedImage(aPath.string());
			} catch (std::exception &) {
				image = loadItkImage(aPath, aProgressNotifier.subTaskNotifier(1));
			}
		}
	}

	/*float step = 5.0f / 128;
	M4D::Imaging::Image< Vector<uint16, 3 >, 3 >::Ptr image =
		M4D::Imaging::ImageFactory::CreateEmptyImage3DTyped< Vector<uint16, 3 > >( 128, 128, 32 );
	for( unsigned i=0; i<128; ++i ) {
		for( unsigned j=0; j<128; ++j ) {
			for( unsigned k=0; k<32; ++k ) {
				float x = (i - 64) * step;
				float y = (j - 64) * step;
				image->GetElement( Vector< int32, 3 >( i, j, k ) ) = Vector<uint16, 3 >(
							(i % 30) * 40,
							(j % 60) * 30,
							(k * 80)
							);
			}
		}
	}*/
	return image;
}


DatasetID
DatasetManager::registerDataset(M4D::Imaging::AImage::Ptr aImage, const std::string &aName)
{
	ImageRecord currentRecord;
	currentRecord.assignImage(aImage, aName);
	DatasetID id = addNewRecord(std::move(currentRecord));
	emit registeredNewDataset(id);
	return id;
}

void
DatasetManager::closeAll()
{
	mImageModel.clear();
}

std::shared_ptr<ImageStatistics>
DatasetManager::getImageStatistics(DatasetID aId)
{
	auto & rec = getDatasetRecord(aId);

	/*if (!rec.mStatistics) {
		M4D::Imaging::Histogram1D<int> histogram1D;
		ImageStatistics::ScatterPlot gradientScatterPlot;
		IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( rec.mImage,
			histogram1D = M4D::Imaging::createHistogramForImageRegion2<M4D::Imaging::Histogram1D<int>, IMAGE_TYPE >( IMAGE_TYPE::Cast(*rec.mImage));
			//gradientScatterPlot = M4D::Imaging::createGradientScatterPlotForImageRegion<
			//	ImageStatistics::ScatterPlot,
			//	IMAGE_TYPE::SubRegion>(IMAGE_TYPE::Cast( *rec.mImage ).GetRegion());
		);
		auto statistics = std::make_shared<ImageStatistics>();
		statistics->mHistogram = std::move(histogram1D);
		statistics->mGradientScatterPlot = std::move(gradientScatterPlot);
		rec.mStatistics = statistics;
	}*/
	return rec.mStatistics;
}

template <typename TImageType1, typename TImageType2>
std::shared_ptr<ImageStatistics>
DatasetManager::getCombinedStatisticsType1Type2(const TImageType1 &aImage1, const TImageType2 &aImage2)
{
	/*auto scatterPlot = M4D::Imaging::createGradientScatterPlotForImageRegionPair<
			ImageStatistics::ScatterPlot,
			typename TImageType1::SubRegion,
			typename TImageType2::SubRegion>(aImage1.GetRegion(), aImage2.GetRegion());
*/
	auto statistics = std::make_shared<ImageStatistics>();
	//statistics->mGradientScatterPlot = std::move(scatterPlot);
	return statistics;
}

template <typename TImageType1>
std::shared_ptr<ImageStatistics>
DatasetManager::getCombinedStatisticsType1(const TImageType1 &aImage1, const M4D::Imaging::AImage &aImage2)
{
	IMAGE_NUMERIC_TYPE_CONST_REF_SWITCH_MACRO_3D(aImage2,
		return getCombinedStatisticsType1Type2(aImage1, IMAGE);
	);
	return std::shared_ptr<ImageStatistics>();
}

std::shared_ptr<ImageStatistics>
DatasetManager::getCombinedStatistics(DatasetID aPrimaryId, DatasetID aSecondaryId)
{
//TODO
	auto &rec1 = getDatasetRecord(aPrimaryId);
	auto &rec2 = getDatasetRecord(aSecondaryId);
	IMAGE_NUMERIC_TYPE_CONST_REF_SWITCH_MACRO_3D(*rec1.mImage,
		return getCombinedStatisticsType1<IMAGE_TYPE>(IMAGE, *rec2.mImage);
			);
	return std::shared_ptr<ImageStatistics>();
}

void DatasetManager::saveDataset(DatasetID aId, boost::filesystem::path aPath)
{
	try {
		prognot::qt::ProgressDialog progressDialog(QApplication::activeWindow());
		//TODO QString to path proper conversion on win
		auto & rec = getDatasetRecord(aId);
		if (!rec.mImage) {
			M4D_THROW(EItemNotFound());
		}
		auto image = rec.mImage;

		// TODO when compilers fix passing movable types to async
		std::shared_ptr<prognot::ProgressNotifier> notifierPtr(new prognot::ProgressNotifier(std::move(progressDialog.progressNotifier())));
		std::async(
			std::launch::async,
			[aPath, image, notifierPtr] () {
				saveItkImage(*image, aPath, std::move(*notifierPtr));
			});
		progressDialog.exec();
	} catch ( std::exception &e ) {
		QMessageBox::critical ( NULL, "Exception", QString( e.what() ) );
	}
	STUBBED("Implement dataset save!");

}

int ImageListModel::rowCount(const QModelIndex &parent) const
{
	return int(mDatasetIDList.size());
}

QVariant
ImageListModel::data(const QModelIndex &index, int role) const
{
	if (role != Qt::DisplayRole) {
		return QVariant();
	}
	if (index.row() < int(mDatasetIDList.size())) {
		const auto &rec = getDatasetRecord(mDatasetIDList[index.row()]);
		switch (index.column()) {
		case 0:
			return QVariant(int(idFromIndex(index.row())));
		case 1:
			return QVariant(rec.name().c_str());
		case 2:
			return QVariant(rec.path().c_str());
		}
	}
	return QVariant();
}

QVariant
ImageListModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role != Qt::DisplayRole || orientation != Qt::Horizontal) {
		return QVariant();
	}

	switch (section) {
	case 0:
		return QVariant("Id");
	case 1:
		return QVariant("Name");
	case 2:
		return QVariant("Path");
	}

	return QVariant();
}



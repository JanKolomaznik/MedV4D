#include "DatasetManager.hpp"

#include <QApplication>
#include <QMessageBox>
#include <future>
#include <memory>

#include "MedV4D/Imaging/ImageFactory.h"
#include "MedV4D/DICOMInterface/DcmProvider.h"

#include <prognot/Qt/ProgressDialog.hpp>


DatasetManager::DatasetID DatasetManager::loadFromFile()
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

		DatasetID currentID = newID();
		auto & currentRecord = mImages[currentID];
		currentRecord.assignImage(image.get());
		return currentID;
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
	M4D::Imaging::AImage::Ptr image;
	if ( aPath.extension() == ".dcm" || aPath.extension() == ".DCM" ) {
		M4D::Dicom::DicomObjSetPtr dicomObjSet = M4D::Dicom::DicomObjSetPtr( new M4D::Dicom::DicomObjSet() );
		M4D::Dicom::DcmProvider::LoadSerieThatFileBelongsTo( aPath, aPath.parent_path(), *dicomObjSet, std::move(aProgressNotifier));
		image = M4D::Dicom::DcmProvider::CreateImageFromDICOM( dicomObjSet );
	} else {
		image = M4D::Imaging::ImageFactory::LoadDumpedImage( aPath.string() );
	}
	return image;
}

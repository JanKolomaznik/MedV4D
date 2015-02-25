#include "ExtendedViewerControls.hpp"
#include "ui_ExtendedViewerControls.h"
#include "itkUtils.hpp"

#include "ItkFiltering.h"
#include "EigenvaluesFilterPolicies.h"

ExtendedViewerControls::ExtendedViewerControls(DatasetManager &aManager, QWidget *parent)
  : QWidget(parent)
  , ui(new Ui::ExtendedViewerControls)
  , mManager(aManager),
  mIsUpdating(false)
{
  ui->setupUi(this);

  ui->mPrimaryDatasetComboBox->setModel(&mManager.imageModel());
  ui->mSecondaryDatasetComboBox->setModel(&mManager.imageModel());
  ui->mMaskComboBox->setModel(&mManager.imageModel());
}

ExtendedViewerControls::~ExtendedViewerControls()
{
  delete ui;
}

ViewerControls &
ExtendedViewerControls::viewerControls() const
{
  return *(ui->mViewerControls);
}

void
ExtendedViewerControls::setViewer(M4D::GUI::Viewer::GeneralViewer *aViewer)
{
  viewerControls().setViewer(aViewer);
  updateAssignedDatasets();
}

void
ExtendedViewerControls::updateControls()
{
  updateAssignedDatasets();
  viewerControls().updateControls();
}

void
ExtendedViewerControls::createDataset()
{
  int primaryIndex = ui->mPrimaryDatasetComboBox->currentIndex();

  if (primaryIndex >= 0) {
    auto inputData = ViewerInputDataWithId::Ptr(new ViewerInputDataWithId);
    auto id = mManager.idFromIndex(primaryIndex);
    const auto & rec = mManager.getDatasetRecord(id);

    // rewrite generically!!
    typedef unsigned short ImageElementType;
    auto itkImage = M4dImageToItkImage<ImageElementType>(std::static_pointer_cast<const M4D::Imaging::Image<ImageElementType, 3>>(rec.mImage));

    using namespace M4D::GUI::Viewer;
    ItkFiltering<FranghiVesselness<>> filtering(itkImage);

    auto filteredImage = filtering.GetEigenValuesFilterImage();

    auto medV4DImage = itkImageToM4dImage<ImageElementType>(itkImage);
    this->mManager.registerDataset(medV4DImage, "itkConvertedImage");
    this->updateAssignedDatasets();
  }
}

void
ExtendedViewerControls::updateAssignedDatasets()
{
  if (mIsUpdating) {
    return;
  }
  if (!viewerControls().viewer()) {
    setEnabled(false);
    return;
  }
  setEnabled(true);
  auto input = viewerControls().viewer()->inputData();
  if (!input) {
    return;
  }
  mIsUpdating = true;
  ViewerInputDataWithId::ConstPtr inputWithId = std::static_pointer_cast<const ViewerInputDataWithId>(input);

  DatasetManager::DatasetID primaryID = inputWithId->primaryImageId();
  DatasetManager::DatasetID secondaryID = inputWithId->secondaryImageId();
  DatasetManager::DatasetID maskID = inputWithId->maskId();

  int primaryIndex = mManager.indexFromID(primaryID);
  int secondaryIndex = mManager.indexFromID(secondaryID);
  int maskIndex = mManager.indexFromID(maskID);

  ui->mPrimaryDatasetComboBox->setCurrentIndex(primaryIndex);
  ui->mSecondaryDatasetComboBox->setCurrentIndex(secondaryIndex);
  ui->mMaskComboBox->setCurrentIndex(maskIndex);

  mIsUpdating = false;
}

void ExtendedViewerControls::assignDatasets()
{
  if (mIsUpdating) {
    return;
  }
  if (!viewerControls().viewer()) {
    return;
  }

  int primaryIndex = ui->mPrimaryDatasetComboBox->currentIndex();
  int secondaryIndex = ui->mSecondaryDatasetComboBox->currentIndex();
  int maskIndex = ui->mMaskComboBox->currentIndex();

  auto inputData = ViewerInputDataWithId::Ptr(new ViewerInputDataWithId);

  if (primaryIndex >= 0) {
    auto id = mManager.idFromIndex(primaryIndex);
    try {
      const auto & rec = mManager.getDatasetRecord(id);
      inputData->assignPrimaryImageWithId(std::static_pointer_cast<M4D::Imaging::AImageDim<3>>(rec.mImage), id);
    }
    catch (std::exception &) {
      D_PRINT("Dataset with id : " << id << " not found!");
    }
  }

  if (secondaryIndex >= 0) {
    auto id = mManager.idFromIndex(secondaryIndex);
    try {
      const auto & rec = mManager.getDatasetRecord(id);
      inputData->assignSecondaryImageWithId(std::static_pointer_cast<M4D::Imaging::AImageDim<3>>(rec.mImage), id);
    }
    catch (std::exception &) {
      D_PRINT("Dataset with id : " << id << " not found!");
    }
  }

  if (maskIndex >= 0) {
    auto id = mManager.idFromIndex(maskIndex);
    try {
      const auto & rec = mManager.getDatasetRecord(id);
      inputData->assignMaskWithId(std::static_pointer_cast<M4D::Imaging::AImageDim<3>>(rec.mImage), id);
    }
    catch (std::exception &) {
      D_PRINT("Mask dataset with id : " << id << " not found!");
    }
  }

  viewerControls().viewer()->setInputData(inputData);
}

void ExtendedViewerControls::resetPrimaryDataset()
{
  ui->mPrimaryDatasetComboBox->setCurrentIndex(-1);
}

void ExtendedViewerControls::resetSecondaryDataset()
{
  ui->mSecondaryDatasetComboBox->setCurrentIndex(-1);
}

void ExtendedViewerControls::resetMaskDataset()
{
  ui->mMaskComboBox->setCurrentIndex(-1);
}

#include "ExtendedViewerControls.hpp"
#include "ui_ExtendedViewerControls.h"
#include "ItkUtils.hpp"

#include "ItkFiltering.h"
#include "ItkEigenvalues.h"
#include "EigenvaluesFilterPolicies.h"

#include "AdditionalDatasetOptions.h"

#include <vector>
#include <string>


using namespace M4D::GUI::Viewer;

std::vector<std::string> additionalDatasetDropdownBox =
{
  "Eigenvalues raw",
  "Eigenvalues linear combination",
  "Franghi's vesselness",
  "Sato's vesselness",
  "Laplacian of Gaussian"
};

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

  this->ui->createDatasetTypeComboBox->clear();
  for (auto additionalDatasetName : additionalDatasetDropdownBox)
  {
    this->ui->createDatasetTypeComboBox->addItem(QString::fromStdString(additionalDatasetName));
  }
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
  // rewrite generically!!
  typedef unsigned short ImageElementType;
  const size_t DIMENSION = 3;

  int primaryIndex = ui->mPrimaryDatasetComboBox->currentIndex();

  if (primaryIndex >= 0) {

    size_t selectedPolicyIndex = this->ui->createDatasetTypeComboBox->currentIndex();
    AdditionalDatasetOptions datasetFunctionOptions;

    if (datasetFunctionOptions.exec() == QDialog::Accepted)
    {
      auto inputData = ViewerInputDataWithId::Ptr(new ViewerInputDataWithId);
      auto id = mManager.idFromIndex(primaryIndex);
      const auto & rec = mManager.getDatasetRecord(id);

      auto itkImage = M4dImageToItkImage<ImageElementType>(std::static_pointer_cast<const M4D::Imaging::Image<ImageElementType, DIMENSION>>(rec.mImage));

      std::vector<EigenvalueType> options = datasetFunctionOptions.GetValues();

      switch (selectedPolicyIndex)
      {
      case 0:
      {
        ItkEigenvalues<ImageElementType, float> eigenvaluesComputer(itkImage, options[0]);
        itk::Image<Vector<float, DIMENSION>, DIMENSION>::Pointer filteredImage = eigenvaluesComputer.GetEigenValuesImage();
        M4D::Imaging::Image<Vector<float, DIMENSION>, DIMENSION>::Ptr medV4DImage = itkImageToM4dImage<Vector<float, DIMENSION>>(filteredImage);
        this->mManager.registerDataset(medV4DImage, additionalDatasetDropdownBox[selectedPolicyIndex]);
      }
      break;

      case 1:
      {
        typedef EigenvaluesLinearCombination<ImageElementType, EigenvalueType, DIMENSION> LinearCombinationType;
        LinearCombinationType linearCombination(options);
        ItkFiltering<LinearCombinationType, ImageElementType> filtering(itkImage, linearCombination);
        itk::Image<ImageElementType, DIMENSION>::Pointer filteredImage = filtering.GetEigenValuesFilterImage();
        auto medV4DImage = itkImageToM4dImage<ImageElementType>(filteredImage);
        this->mManager.registerDataset(medV4DImage, additionalDatasetDropdownBox[selectedPolicyIndex]);
      }
      break;

      case 2:
      {
        typedef FranghiVesselness<ImageElementType, EigenvalueType, DIMENSION> VesselnessType;
        VesselnessType vesselness(options);
        ItkFiltering<VesselnessType, ImageElementType> filtering(itkImage, vesselness);
        itk::Image<ImageElementType, DIMENSION>::Pointer filteredImage = filtering.GetEigenValuesFilterImage();
        auto medV4DImage = itkImageToM4dImage<ImageElementType>(filteredImage);
        this->mManager.registerDataset(medV4DImage, additionalDatasetDropdownBox[selectedPolicyIndex]);
      }
      break;

      case 3:
      {
        typedef ParameterPolicy<ImageElementType, EigenvalueType, DIMENSION> EmptyPolicyType;
        EmptyPolicyType emptyPolicy(options);
        ItkFiltering<EmptyPolicyType, ImageElementType> filtering(itkImage, emptyPolicy);
        itk::Image<ImageElementType, DIMENSION>::Pointer filteredImage = filtering.GetSatoVesselnessFilterImage(emptyPolicy.alpha, emptyPolicy.beta);
        auto medV4DImage = itkImageToM4dImage<ImageElementType>(filteredImage);
        this->mManager.registerDataset(medV4DImage, additionalDatasetDropdownBox[selectedPolicyIndex]);
      }
      break;

      case 4:
      {
        typedef ParameterPolicy<ImageElementType, EigenvalueType, DIMENSION> EmptyPolicyType;
        EmptyPolicyType emptyPolicy(options);
        ItkFiltering<EmptyPolicyType, ImageElementType> filtering(itkImage, emptyPolicy);
        itk::Image<ImageElementType, DIMENSION>::Pointer filteredImage = filtering.GetLaplacianOfGaussianFilterImage();
        auto medV4DImage = itkImageToM4dImage<ImageElementType>(filteredImage);
        this->mManager.registerDataset(medV4DImage, additionalDatasetDropdownBox[selectedPolicyIndex]);
      }
      break;
      }


      this->updateAssignedDatasets();
    }

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

  this->ui->createDatasetPushButton->setEnabled(primaryIndex >= 0);
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

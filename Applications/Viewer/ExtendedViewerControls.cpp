#include "ExtendedViewerControls.hpp"
#include "ui_ExtendedViewerControls.h"

ExtendedViewerControls::ExtendedViewerControls(DatasetManager &aManager, QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::ExtendedViewerControls)
	, mManager(aManager),
	  mIsUpdating(false)
{
	ui->setupUi(this);

	ui->mPrimaryDatasetComboBox->setModel(&mManager.imageModel());
	ui->mSecondaryDatasetComboBox->setModel(&mManager.imageModel());
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

	int primaryIndex = mManager.indexFromID(primaryID);
	int secondaryIndex = mManager.indexFromID(secondaryID);

	ui->mPrimaryDatasetComboBox->setCurrentIndex(primaryIndex);
	ui->mSecondaryDatasetComboBox->setCurrentIndex(secondaryIndex);

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

	auto inputData = ViewerInputDataWithId::Ptr(new ViewerInputDataWithId);

	if (primaryIndex >= 0) {
		auto id = mManager.idFromIndex(primaryIndex);
		try {
			const auto & rec = mManager.getDatasetRecord(id);
			inputData->assignPrimaryImageWithId(std::static_pointer_cast<M4D::Imaging::AImageDim<3>>(rec.mImage), id);
		} catch (std::exception &) {
			D_PRINT("Dataset with id : " << id << " not found!");
		}
	}

	if (secondaryIndex >= 0) {
		auto id = mManager.idFromIndex(secondaryIndex);
		try {
			const auto & rec = mManager.getDatasetRecord(id);
			inputData->assignSecondaryImageWithId(std::static_pointer_cast<M4D::Imaging::AImageDim<3>>(rec.mImage), id);
		} catch (std::exception &) {
			D_PRINT("Dataset with id : " << id << " not found!");
		}
	}

	viewerControls().viewer()->setInputData(inputData);
}

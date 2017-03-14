#include "ExtendedViewerControls.hpp"
#include "ui_ExtendedViewerControls.h"

ExtendedViewerControls::ExtendedViewerControls(DatasetManager &aManager, QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::ExtendedViewerControls)
	, mManager(aManager)
	, mIsUpdating(false)
{
	ui->setupUi(this);

	ui->mPrimaryDatasetComboBox->setModel(&mManager.imageModel());
	ui->mSecondaryDatasetComboBox->setModel(&mManager.imageModel());
	ui->mMaskComboBox->setModel(&mManager.imageModel());

	ui->mPrimaryDatasetComboBox->setModelColumn(1);
	ui->mSecondaryDatasetComboBox->setModelColumn(1);
	ui->mMaskComboBox->setModelColumn(1);

	ui->mReferencedViewer->setModel(ViewerManager::getInstance()->registeredViewers());

	/*QObject::connect(
			ui->mPrimaryDatasetComboBox->model(),
			&QAbstractItemModel::modelReset,
				[this]() {
					QMessageBox::information(nullptr, "My Application", "reset item list");
					ui->mPrimaryDatasetComboBox->setModel(&mManager.imageModel());
			});*/
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
	updateViewerConnection();
}

void ExtendedViewerControls::updateViewerConnection()
{
	if (mIsUpdating) {
		return;
	}
	if (!viewerControls().viewer()) {
		return;
	}
	auto masterViewer = viewerControls().viewer()->mMasterViewer;
	ui->mReferencedViewer->setCurrentIndex(ViewerManager::getInstance()->getViewerIndex(masterViewer));
}

void ExtendedViewerControls::setViewerConnection()
{
	if (!viewerControls().viewer()) {
		return;
	}
	viewerControls().viewer()->followViewer(getSelectedMasterViewer());
}

M4D::GUI::Viewer::GeneralViewer *
ExtendedViewerControls::getSelectedMasterViewer()
{
	return static_cast<M4D::GUI::Viewer::GeneralViewer *>(ViewerManager::getInstance()->getViewer(ui->mReferencedViewer->currentIndex()));
}

void ExtendedViewerControls::clearViewerConnection()
{
	ui->mReferencedViewer->setCurrentIndex(-1);
}

void
ExtendedViewerControls::updateControls()
{
	updateAssignedDatasets();
	viewerControls().updateControls();
	updateViewerConnection();
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

	DatasetID primaryID = inputWithId->primaryImageId();
	DatasetID secondaryID = inputWithId->secondaryImageId();
	DatasetID maskID = inputWithId->maskId();

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

	if (maskIndex >= 0) {
		auto id = mManager.idFromIndex(maskIndex);
		try {
			const auto & rec = mManager.getDatasetRecord(id);
			inputData->assignMaskWithId(std::static_pointer_cast<M4D::Imaging::AImageDim<3>>(rec.mImage), id);
		} catch (std::exception &) {
			D_PRINT("Mask dataset with id : " << id << " not found!");
		}
	}

	viewerControls().viewer()->setInputData(inputData, ui->mResetView->isChecked());
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

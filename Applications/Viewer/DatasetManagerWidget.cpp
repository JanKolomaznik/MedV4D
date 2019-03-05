#include "DatasetManagerWidget.hpp"
#include "ui_DatasetManagerWidget.h"

#include <QMenu>

DatasetManagerWidget::DatasetManagerWidget(DatasetManager &aManager, QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::DatasetManagerWidget)
	, mManager(aManager)
	, mExportFileDialog(nullptr, tr("Export Dataset"))
{
	ui->setupUi(this);
	ui->mDatasetView->setModel(&mManager.imageModel());

	mExportFileDialog.setAcceptMode(QFileDialog::AcceptSave);

	QStringList filters;
	filters << "Dump files (*.dump)"
		<< "MRC files (*.mrc *.MRC)";
		//<< "Any files (*)";
	mExportFileDialog.setNameFilters(filters);
}

DatasetManagerWidget::~DatasetManagerWidget()
{
	delete ui;
}

void
DatasetManagerWidget::datasetViewContextMenu(const QPoint &pos)
{
	QPoint globalpos = ui->mDatasetView->mapToGlobal(pos);

	//QMenu menuBeyondItem;
	//QAction* action_addElement = menuBeyondItem.addAction("Add");

	QMenu menuForItem;
	QAction* actionExport = menuForItem.addAction("Export...");
	QAction* purgeFromGPUExport = menuForItem.addAction("Purge from GPU");

	QModelIndex pointedIndex = ui->mDatasetView->indexAt(pos);
	auto datasetId = mManager.idFromIndex(pointedIndex.row());

	purgeFromGPUExport->setEnabled(mManager.isOnGPU(datasetId));
	QAction* selectedAction;
	/*if(!pointedItem) {
		selectedAction = menuBeyondItem.exec(globalpos);
		if(selectedAction) {
			if(selectedAction == action_addElement) {
				qDebug() << "Add";
			}
		}
	}
	else*/ {
		selectedAction = menuForItem.exec(globalpos);
		if(selectedAction) {
			if(selectedAction == actionExport) {
				if (mExportFileDialog.exec()) {
					auto files = mExportFileDialog.selectedFiles();
					if (!files.empty()) {
						boost::filesystem::path path = files.front().toStdString();
						mManager.saveDataset(datasetId, path);
					}
				}
			}
			if (selectedAction == purgeFromGPUExport) {
				mManager.purgeFromGPU(datasetId);
			}
		}
	}
}

#ifndef DATASETMANAGERWIDGET_HPP
#define DATASETMANAGERWIDGET_HPP

#include "DatasetManager.hpp"

#include <QWidget>
#include <QFileDialog>

namespace Ui {
class DatasetManagerWidget;
}

class DatasetManagerWidget : public QWidget
{
	Q_OBJECT

public:
	explicit DatasetManagerWidget(DatasetManager &aManager, QWidget *parent = nullptr);
	~DatasetManagerWidget();

public slots:
	void
	datasetViewContextMenu(const QPoint &pos);
private:
	Ui::DatasetManagerWidget *ui;

	DatasetManager &mManager;


	QFileDialog mExportFileDialog;
};

#endif // DATASETMANAGERWIDGET_HPP

#ifndef EXTENDEDVIEWERCONTROLS_HPP
#define EXTENDEDVIEWERCONTROLS_HPP

#include <QWidget>
#include "MedV4D/GUI/widgets/ViewerControls.h"

#include "DatasetManager.hpp"

namespace Ui {
class ExtendedViewerControls;
}

class ExtendedViewerControls : public QWidget
{
	Q_OBJECT

public:
	explicit ExtendedViewerControls(DatasetManager &aManager, QWidget *parent = 0);
	~ExtendedViewerControls();

	ViewerControls &
	viewerControls() const;

	void
	setViewer(M4D::GUI::Viewer::GeneralViewer *aViewer);

public slots:
	void
	updateControls();

	void
	updateAssignedDatasets();

	void
  assignDatasets();

  void
  createDataset();

	void
	resetPrimaryDataset();
	void
	resetSecondaryDataset();
	void
	resetMaskDataset();
private:
	Ui::ExtendedViewerControls *ui;

	DatasetManager &mManager;

	bool mIsUpdating;
};

#endif // EXTENDEDVIEWERCONTROLS_HPP

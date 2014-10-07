#ifndef EXTENDEDVIEWERCONTROLS_HPP
#define EXTENDEDVIEWERCONTROLS_HPP

#include <QWidget>
#include "MedV4D/GUI/widgets/ViewerControls.h"

namespace Ui {
class ExtendedViewerControls;
}

class ExtendedViewerControls : public QWidget
{
	Q_OBJECT

public:
	explicit ExtendedViewerControls(QWidget *parent = 0);
	~ExtendedViewerControls();

	ViewerControls &
	viewerControls() const;
private:
	Ui::ExtendedViewerControls *ui;
};

#endif // EXTENDEDVIEWERCONTROLS_HPP

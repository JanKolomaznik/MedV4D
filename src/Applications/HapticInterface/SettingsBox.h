#ifndef M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_HAPTIC_VIEWER_SETTINGSBOX
#define M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_HAPTIC_VIEWER_SETTINGSBOX

#include <QtGui/QWidget>
#include "ViewerWindow.h"

namespace Ui
{
	class SettingsBox;
}

class ViewerWindow;

class SettingsBox : public QWidget
{
	Q_OBJECT

public:
	SettingsBox(ViewerWindow* parent);
	~SettingsBox();

	Ui::SettingsBox* ui;

signals:
	void scaleChanged(double scale);

protected:
	ViewerWindow* parent;

protected slots:
	void slotChangeScale();
};

#endif
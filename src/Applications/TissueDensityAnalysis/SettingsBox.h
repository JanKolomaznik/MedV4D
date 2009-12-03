#ifndef SETTINGSBOX
#define SETTINGSBOX

#include <QtGui/QWidget>
#include "GUI/widgets/m4dGUIMainWindow.h"
#include "m4dMySliceViewerWidget.h"
#include <map>

using namespace std;
namespace Ui{

    class SettingsBox;
}

class SettingsBox : public QWidget{

    Q_OBJECT

public:
	SettingsBox(M4D::GUI::m4dGUIMainWindow * parent);
    ~SettingsBox();

	virtual void build();

    Ui::SettingsBox* ui;

protected slots:
	void slotSetSphereCenter(double x, double y, double z);
	void slotSetSphereRadius(int amountA, int amountB, double zoomRate);

protected:
	M4D::GUI::m4dGUIMainWindow *_parent;

};

#endif //SETTINGSBOX
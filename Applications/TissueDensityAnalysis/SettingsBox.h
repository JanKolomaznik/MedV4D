#ifndef SETTINGSBOX
#define SETTINGSBOX

#include <QtGui/QWidget>
#include "MedV4D/GUI/widgets/m4dGUIMainWindow.h"
#include "TDASliceViewerWidget.h"
#include "TDASphereSelection.h"
#include <map>


typedef int16	ElementType;
const unsigned Dim = 3;
typedef M4D::Imaging::Image< ElementType, Dim > ImageType;

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
	void setMaskFilter(M4D::Imaging::TDASphereSelection< ImageType >	*filter);

protected slots:
	void slotSetSphereCenter(double x, double y, double z);
	void slotSetSphereRadius(int amountA, int amountB, double zoomRate);
	void slotCreateMask();

protected:
	M4D::GUI::m4dGUIMainWindow *_parent;
	M4D::Imaging::TDASphereSelection< ImageType >	*_filter;
	

	

};

#endif //SETTINGSBOX
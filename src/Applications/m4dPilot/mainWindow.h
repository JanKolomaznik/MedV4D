#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "GUI/m4dGUIMainWindow.h"

#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "m4dPilot_0"


class mainWindow: public M4D::GUI::m4dGUIMainWindow
{
	Q_OBJECT

public:

	mainWindow ();

protected:
	void
	process ( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjSet );

	M4D::Imaging::AbstractImageConnection _conn;
private:

};


#endif // MAIN_WINDOW_H



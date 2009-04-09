#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "GUI/widgets/m4dGUIMainWindow.h"

#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "m4dPilot"


class mainWindow: public M4D::GUI::m4dGUIMainWindow
{
	Q_OBJECT

public:

	mainWindow ();

protected:

	void process ( M4D::Imaging::AbstractDataSet::Ptr inputDataSet );
};


#endif // MAIN_WINDOW_H



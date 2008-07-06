#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "GUI/m4dGUIMainWindow.h"

#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "m4dPilot"


class mainWindow: public m4dGUIMainWindow
{
  Q_OBJECT

  public:

    mainWindow ();

  private:

    void view ( M4D::Dicom::DcmProvider::DicomObjSet *dicomObjSet );
};


#endif // MAIN_WINDOW_H



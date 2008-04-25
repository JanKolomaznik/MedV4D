#include <QApplication>

#include "../widgets/m4dGUIMainWindow.h"

#include "../../include/M4DCommon.h"

// not a final solution
#ifdef OS_WIN
  #include "projectConfig.h"
#endif


int main ( int argc, char *argv[] )
{
  QApplication app( argc, argv );
  app.setQuitOnLastWindowClosed( true );

  m4dGUIMainWindow mainWindow;
  mainWindow.show();

  return app.exec();
}
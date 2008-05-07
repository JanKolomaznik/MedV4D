#include <QApplication>

#include "../widgets/m4dGUIMainWindow.h"

#include "Common.h"


int main ( int argc, char *argv[] )
{
  QApplication app( argc, argv );
  app.setQuitOnLastWindowClosed( true );

  m4dGUIMainWindow mainWindow;
  mainWindow.show();

  return app.exec();
}

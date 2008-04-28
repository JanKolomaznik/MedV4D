#include <QtGui/QtGui>

#include "qtVtkMainWindow.h"

#include "../../include/M4DCommon.h"


int main ( int argc, char *argv[] )
{
  QApplication app( argc, argv );
  app.setQuitOnLastWindowClosed( true );

  MainWindow mainWin;
  mainWin.show();

  return app.exec();
}
#include <QtGui/QtGui>

#include "uiPureQtMainWindow.h"


int main( int argc, char **argv ) 
{
  QApplication app( argc, argv );
  app.setQuitOnLastWindowClosed( true );

  QMainWindow *form = new QMainWindow;
  Ui::MainWindow ui;
  ui.setupUi( form );

  ui.textBrowser->setSource( QUrl( "..\\..\\GUI\\pureQt\\main.cpp" ) );


  form->show();
  return app.exec();
}
#include <QtGui/QtGui>

#include "ui_browser.h"


int main( int argc, char **argv ) 
{
  QApplication app( argc, argv );
  app.setQuitOnLastWindowClosed( true );

  QMainWindow *form = new QMainWindow;
  Ui::MainWindow ui;
  ui.setupUi( form );

  ui.textBrowser->setSource( QUrl( "main.cpp" ) );


  form->show();
  return app.exec();
}
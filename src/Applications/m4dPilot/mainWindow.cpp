#include "mainWindow.h"

using namespace std;


mainWindow::mainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME )
{
  Q_INIT_RESOURCE( mainWindow ); 

  QCoreApplication::setOrganizationName( ORGANIZATION_NAME );
  QCoreApplication::setApplicationName( APPLICATION_NAME );

  // createPipeline();

  // tell mainWindow about possible connections - can be during the creation of pipeline (connections)

  // M4D::Imaging::ConnectionInterface *conn;
  // addSource( conn, "Bone segmentation", "Stage #1" );
  // addSource( conn, "Bone segmentation", "Result" );
  
  // add your own settings widgets

  // addDockWindow( "Bone Segmentation", new QListWidget );
}


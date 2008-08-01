#include "mainWindow.h"

using namespace std;


mainWindow::mainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME )
{
  Q_INIT_RESOURCE( mainWindow ); 

  QCoreApplication::setOrganizationName( ORGANIZATION_NAME );
  QCoreApplication::setApplicationName( APPLICATION_NAME );

  // createPipeline();
  // tell mainWindow about possible connections - can be during creating the pipeline (connections)
  // M4D::Imaging::ConnectionInterface *conn;
  // mainViewerDesktop->addSource( conn, "Bone segmentation", "Stage #1" );
  // mainViewerDesktop->addSource( conn, "Bone segmentation", "Result" );
}


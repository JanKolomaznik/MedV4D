/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file LoadingThreads.h
 * @{ 
 **/

#ifndef LOADING_THREADS_H
#define LOADING_THREADS_H

#include <QObject>

#include "backendForDICOM/DICOMServiceProvider.h"


namespace M4D {
namespace GUI {

class OpenLoadingThread: public QObject 
{ 
  Q_OBJECT

  public:

    OpenLoadingThread ( std::string fileName, std::string folder, Dicom::DicomObjSet *result,
                        QObject *mainWindow )
      : fileName( fileName ), folder( folder ), result( result ), mainWindow( mainWindow )
    {} 

    OpenLoadingThread ( const OpenLoadingThread &lt )
    {
      fileName   = lt.fileName;
      folder     = lt.folder;
      result     = lt.result;
      mainWindow = lt.mainWindow;

      connect( this, SIGNAL( ready() ), mainWindow, SLOT( loadingReady() ), Qt::QueuedConnection );
      connect( this, SIGNAL( exception( const QString & ) ), mainWindow, SLOT( loadingException( const QString & ) ), 
               Qt::QueuedConnection );
    }

    void operator() ();

  signals:

    void ready ();

    void exception ( const QString &description );

  private:

    std::string fileName, folder;
    Dicom::DicomObjSet *result;
    QObject *mainWindow;
};

} // namespace GUI
} // namespace M4D

#endif // LOADING_THREADS_H


/** @} */


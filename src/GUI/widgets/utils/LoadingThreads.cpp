/**
 *  @ingroup gui
 *  @file LoadingThreads.cpp
 *  @brief some brief
 */
#include "GUI/LoadingThreads.h"

using namespace M4D::Dicom;


namespace M4D {
namespace GUI {

void OpenLoadingThread::operator() ()
{
  try {

    DcmProvider::LoadSerieThatFileBelongsTo( fileName, folder, *result );

    emit ready();

  }
  catch ( ErrorHandling::ExceptionBase &e ) {
    emit exception( e.what() );
  }
}

} // namespace GUI
} // namespace M4D


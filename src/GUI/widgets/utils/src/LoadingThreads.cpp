/**
 *  @ingroup gui
 *  @file LoadingThreads.cpp
 *  @brief some brief
 */
#include "MedV4D/GUI/widgets/utils/LoadingThreads.h"

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


void SearchLoadingThread::operator() ()
{
  if ( isLocal )
  {
    try {

      DcmProvider::LocalGetImageSet( patientID, studyID, serieID, *result );

      emit ready();

    }
    catch ( ErrorHandling::ExceptionBase &e ) {
      emit exception( e.what() );
    }
  }
  else
  {
    DcmProvider::GetImageSet( patientID, studyID, serieID, *result );

    emit ready();  
  }
}

} // namespace GUI
} // namespace M4D


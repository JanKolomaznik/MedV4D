#include "GUI/StudyFilter.h"

using namespace M4D::Dicom;


namespace M4D {
namespace GUI {

void StudyFilter::filter ( DcmProvider::ResultSet *resultSet )
{
  if ( resultSet->empty() ) {
    return;
  }

  for ( unsigned i = 0; i < resultSet->size(); i++ )
  {
    resultSet->erase( resultSet->begin() + 1 );
  }

}

} // namespace GUI
} // namespace M4D
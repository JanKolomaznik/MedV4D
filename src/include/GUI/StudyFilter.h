#ifndef STUDY_FILTER_H
#define STUDY_FILTER_H

// DICOM includes:
#include "Common.h"
#include "dicomConn/DICOMServiceProvider.h"


namespace M4D {
namespace GUI {

/**
 * Class providing study filtering functionality for Study Manager.
 */
class StudyFilter {

  public:

    static void filter ( M4D::Dicom::DcmProvider::ResultSet *resultSet ); 

};

} // namespace GUI
} // namespace M4D

#endif // STUDY_FILTER_H
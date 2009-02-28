/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file StudyFilter.h 
 * @{ 
 **/

#ifndef STUDY_FILTER_H
#define STUDY_FILTER_H

// DICOM includes:
#include "Common.h"
#include "backendForDICOM/DICOMServiceProvider.h"


namespace M4D {
namespace GUI {

/**
 * Class providing client study filtering functionality for Study Manager.
 */
class StudyFilter {

  public:

    /** 
     * Filters ResultSet (TableRows) - filtering modality of studies with given modality vector (checked
     * modalities in Study Manager) - needed after remote find.
     * 
     * @param resultSet ResultSet to filter
     * @param modalitiesVect reference to vector of strings containing set of wanted modalities
     */
    static void filterModalities ( M4D::Dicom::ResultSet *resultSet, 
                                   const M4D::Dicom::StringVector &modalitiesVect ); 

    /** 
     * Filters ResultSet (TableRows) - filtering with all of the Study Manager's input values - 
     * needed after local & recent find.
     * 
     * @param resultSet ResultSet to filter
     * @param firstName reference to string containing first name of the patient
     * @param lastName reference to string containing last name of the patient
     * @param patientID patient ID search mask
     * @param fromDate reference to string containing date (from) in yyyyMMdd format
     * @param toDate reference to string containing date (to) in yyyyMMdd format 
     * @param modalitiesVect reference to vector of strings containing set of wanted modalities
     * @param referringMD reference to string containing referring MD
     * @param description reference to string containing description of the study
     */
    static void filterAll ( M4D::Dicom::ResultSet *resultSet, 
                            const std::string &firstName, const std::string &lastName, 
                            const std::string &patientID, 
                            const std::string &fromDate, const std::string &toDate,
                            const M4D::Dicom::StringVector &modalitiesVect,
                            const std::string &referringMD, const std::string &description ); 

    /** 
     * Filters ResultSet (TableRows) - in ResultSet remains only unique TableRows - all different 
     * from given (currently inserted) one (differs in patientID and studyID - unique key).
     * 
     * @param resultSet ResultSet to filter
     * @param row given TableRow - all TableRows in ResultSet will be different from this
     */
    static void filterDuplicates ( M4D::Dicom::ResultSet *resultSet, 
                                   const M4D::Dicom::TableRow *row );
};

} // namespace GUI
} // namespace M4D

#endif // STUDY_FILTER_H


/** @} */


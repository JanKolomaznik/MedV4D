#include "GUI/StudyFilter.h"

using namespace M4D::Dicom;

using namespace std;


namespace M4D {
namespace GUI {

void StudyFilter::filterModalities ( DcmProvider::ResultSet *resultSet,
                                     const DcmProvider::StringVector &modalitiesVect )
{
  if ( resultSet == 0 ) {
    return;
  }

  for ( unsigned i = 0; i < resultSet->size(); i++ )
  {
    if ( find( modalitiesVect.begin(), modalitiesVect.end(), resultSet->at( i ).modality ) == modalitiesVect.end() ) 
    {
      resultSet->erase( resultSet->begin() + i );
      i--;
    }
  }
}


void StudyFilter::filterAll ( M4D::Dicom::DcmProvider::ResultSet *resultSet, 
                              const std::string &firstName, const std::string &lastName, 
                              const std::string &patientID, 
                              const std::string &fromDate, const std::string &toDate,
                              const M4D::Dicom::DcmProvider::StringVector &modalitiesVect,
                              const std::string &referringMD, const std::string &description )
{
  if ( resultSet == 0 ) {
    return;
  }

  for ( unsigned i = 0; i < resultSet->size(); i++ )
  {
    DcmProvider::TableRow row = resultSet->at( i );

    bool nameMatched = false;
    if ( firstName != "" && lastName == "" ) 
    {
      size_t found;
      found = row.name.find( "_" );
      if ( found == string::npos ) {
        nameMatched = firstName == row.name;
      }
      else {
        nameMatched = firstName == row.name.substr( 0, found );
      }
    }
    else if ( firstName == "" && lastName != "" ) 
    {
      size_t found;
      found = row.name.find( "_" );
      if ( found == string::npos ) {
        nameMatched = lastName == row.name;
      }
      else {
        nameMatched = lastName == row.name.substr( found + 1 );
      }
    } 
    else if ( firstName != "" && lastName != "" ) {
      nameMatched = (firstName + "_" + lastName) == row.name;
    }
    else {
      nameMatched = true;
    }

    bool patientIDMatched   = ( patientID == "" ) ? true : patientID == row.patientID;
    bool fromDateMatched    = ( fromDate == "" ) ? true : atoi( fromDate.c_str() ) <= atoi( row.date.c_str() ); 
    bool toDateMatched      = ( toDate == "" ) ? true : atoi( toDate.c_str() ) >= atoi( row.date.c_str() ); 
    bool modalityMatched    = ( find( modalitiesVect.begin(), modalitiesVect.end(), row.modality ) != 
                                modalitiesVect.end() );
    bool referringMDMatched = ( referringMD == "" ) ? true : referringMD == row.referringMD;
    bool descriptionMatched = ( description == "" ) ? true : description == row.description;

    if ( !nameMatched || !patientIDMatched || !fromDateMatched || !toDateMatched ||
         !modalityMatched || !referringMDMatched || !descriptionMatched ) 
    {
      resultSet->erase( resultSet->begin() + i );
      i--;
    }
  }
}


void StudyFilter::filterDuplicates ( M4D::Dicom::DcmProvider::ResultSet *resultSet, 
                                     const M4D::Dicom::DcmProvider::TableRow *row )
{
  if ( resultSet == 0 ) {
    return;
  }

  string patientID = row->patientID;
  string studyID   = row->studyID;

  for ( unsigned i = 0; i < resultSet->size(); i++ )
  {
    DcmProvider::TableRow currentRow = resultSet->at( i );

    if ( patientID == currentRow.patientID && studyID == currentRow.studyID ) 
    {
      resultSet->erase( resultSet->begin() + i );
      i--;
    }
  }
}

} // namespace GUI
} // namespace M4D


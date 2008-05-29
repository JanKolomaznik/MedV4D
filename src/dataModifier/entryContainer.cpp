#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcfilefo.h>

#include "Common.h"
#include "entryContainer.h"

using namespace M4D::DataModifier;
using namespace std;

///////////////////////////////////////////////////////////////////////

void
EntryContainer::SolveFile( const string &fileName, const string &path)
{
  DcmFileFormat dfile;
  OFCondition cond = dfile.loadFile( fileName.c_str());
  if (! cond.good())
  {
    LOG( "Loading of " << fileName << " failed. ("  << cond.text() << ")" );
    return;
  }

  DcmDataset *dataSet = dfile.getDataset();

  OFString ofStr;
  // edit patient informations
  {
    dataSet->findAndGetOFString( DCM_PatientID, ofStr);

    

    PatientEntry entry( ofStr.c_str(), path);

    PatientInfoMap::iterator found = m_patients.find( entry);
    if( found == m_patients.end() )
    {
      // generate new patInfo and inset it into map
      PatientInfo info;
      m_patients.insert( PatientInfoMap::value_type(
        entry, info) );

      found = m_patients.find( entry);
      m_dict.FillPatientInfo( found->second);
    }

    // i donr know why, but if we have not retrieve data before
    // patient info changing, they will not written down when saving
    {
      const uint16 *data;
      dataSet->findAndGetUint16Array( DCM_PixelData, data);
    }

    PatientInfo *patInfo = &found->second;
    // update patient info in da file
    dataSet->putAndInsertString( 
      DCM_PatientsName, patInfo->patName.c_str(), true);
    dataSet->putAndInsertString( 
      DCM_PatientsBirthDate, patInfo->born.c_str(), true);
    dataSet->putAndInsertString( 
      DCM_PatientsBirthDate, patInfo->born.c_str(), true);
  }

  // edit study informations
  if( ! dateFrom.empty() && ! dateTo.empty() )
  {
    dataSet->findAndGetOFString( DCM_StudyInstanceUID, ofStr);
    string studyID( ofStr.c_str());

    StudyInfoMap::iterator found = m_studies.find( studyID);
    if( found == m_studies.end() )
    {
      // generate new patInfo and inset it into map
      StudyInfo info;
      m_studies.insert( StudyInfoMap::value_type(
        studyID, info) );

      found = m_studies.find( studyID);
      m_dict.GetDateBetween( dateFrom, dateTo, found->second.date);
    }

    StudyInfo *info = &found->second;
    dataSet->putAndInsertString( DCM_StudyDate, info->date.c_str());
  }

  // save the file back
  cond = dfile.saveFile( fileName.c_str() );
  if( cond.bad() )
  {
    LOG( "Saving file: " << fileName << " failed!");
    throw ExceptionBase( "Saving failed");
  }
}

///////////////////////////////////////////////////////////////////////

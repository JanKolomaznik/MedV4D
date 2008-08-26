#include <queue>

#include <dcmtk/config/osconfig.h>
//#include <dcmtk/dcmnet/dimse.h>
//#include <dcmtk/dcmnet/diutil.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcfilefo.h>

// fs
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

#include "Common.h"

#include "dicomConn/DICOMServiceProvider.h"
#include "LocalService.h"

#include "DICOMSupport.h"

namespace fs = boost::filesystem;

using namespace M4D::ErrorHandling;

namespace M4D
{
namespace DicomInternal 
{

///////////////////////////////////////////////////////////////////////

void
LocalService::Find( 
			Dicom::DcmProvider::ResultSet &result,
      const std::string &path)
{
  Reset();

  // save search dir
  m_lastSearchDir = path;

  fs::path full_path = fs::system_complete( fs::path( path) );
  // recursively (through queue) go through all files in subtree
  // of specified directory specified

  try {
    if ( fs::is_directory( full_path ) )
    {
      m_mainQueue.push( full_path);
    }
    else // must be a file
    {
      throw ExceptionBase("Not a folder!");
    }

    fs::path currPath;
    while( ! m_mainQueue.empty() )
    {
      currPath = m_mainQueue.front();
      m_mainQueue.pop();

      SolveDir( currPath, result);
    }
    
  } catch( std::exception &ex) {
    LOG( ex.what());
  }
}

///////////////////////////////////////////////////////////////////////

void
LocalService::FindStudyInfo( 
      DcmProvider::SerieInfoVector &result,
      const std::string &patientID,
			const std::string &studyID)
{
  Entry e;
  e.patID = patientID;
  e.studyID = studyID;

  SetOfEntries::iterator found = m_setOfEntries.find( e);
  if( found == m_setOfEntries.end())
  {
  }
  else
  {
    // copy content of found set into result
    SeriesInStudy *info = &found->second;
    for( SeriesInStudy::iterator i=info->begin(); i != info->end(); i++)
      result.push_back( *i);
  }
}

///////////////////////////////////////////////////////////////////////

void
LocalService::GetImageSet(
      const std::string &patientID,
			const std::string &studyID,
			const std::string &serieID,
			Dicom::DcmProvider::DicomObjSet &result)
{
  fs::path full_path = fs::system_complete( fs::path( m_lastSearchDir) );

  try {
    SolveDirGET( full_path, patientID, studyID, serieID, result);

    fs::path currPath;
    while( ! m_mainQueue.empty() )
    {
      currPath = m_mainQueue.front();
      m_mainQueue.pop();

      SolveDirGET( currPath, patientID, studyID, serieID, result);
    }
    
  } catch( std::exception &ex) {
    LOG( ex.what());
  }
}

///////////////////////////////////////////////////////////////////////

void
LocalService::SolveDir( boost::filesystem::path & dirName,
                       DcmProvider::ResultSet &result)
{
  // Get all files in this dir
  // loop through them
  LOG( "Entering DIRECTORY: " << dirName);

  fs::directory_iterator end_iter;

  for ( fs::directory_iterator dir_itr( dirName );
        dir_itr != end_iter;
        ++dir_itr )
  {
    
    // if it is subdir, call itself on subdir
    if ( fs::is_directory( dir_itr->status() ) )
    {
      m_mainQueue.push( *dir_itr);
    }
    else
    {
      SolveFile( dir_itr->string(), dirName.string(), result );
    }
  }
}

///////////////////////////////////////////////////////////////////////

void
LocalService::SolveFile( 
  const std::string & fileName, const std::string &/* dirName */,
  Dicom::DcmProvider::ResultSet &result)
{
  OFString ofStr;

  DcmFileFormat dfile;
  OFCondition cond = dfile.loadFile( fileName.c_str());
  if (! cond.good())
  {
    LOG( "Loading of " << fileName << " failed. ("  << cond.text() << ")" );
    return;
  }

  DcmDataset *dataSet = dfile.getDataset();

  string setID;
  Entry entry;
  // get info about this file
  {
    dataSet->findAndGetOFString( DCM_PatientID, ofStr);
    entry.patID.append( ofStr.c_str() );

    dataSet->findAndGetOFString( DCM_StudyInstanceUID, ofStr);
    entry.studyID.append( ofStr.c_str() );

    dataSet->findAndGetOFString( DCM_SeriesInstanceUID, ofStr);
    setID.append( ofStr.c_str() );
  }

  // look if it is already in result set
  SetOfEntries::iterator found = m_setOfEntries.find( entry);
  if( found == m_setOfEntries.end() )
  {
    // if not, put new line into result set
    Dicom::DcmProvider::TableRow row;
    GetTableRowFromDataSet( dataSet, &row);
    result.push_back( row);

    SeriesInStudy::value_type item;
    GetSeriesInfo( dataSet, &item);

    // put entry into set
    SeriesInStudy buddy;
    buddy.insert( SeriesInStudy::value_type( item) );

    m_setOfEntries.insert( SetOfEntries::value_type( entry, buddy) );
  }
  else
  {
    SeriesInStudy::value_type item;
    GetSeriesInfo( dataSet, &item);

    // check if setID is already in found record
    SeriesInStudy *foundRecSetIDs = &found->second;
    SeriesInStudy::iterator stud = foundRecSetIDs->find( item);
    
    if( stud == foundRecSetIDs->end() )
    {
      // if not, insert it
      foundRecSetIDs->insert( SeriesInStudy::value_type( item) );
    }
  }
}

///////////////////////////////////////////////////////////////////////

void
LocalService::SolveDirGET( boost::filesystem::path & dirName,
  const std::string &patientID,
	const std::string &studyID,
	const std::string &serieID,
  DcmProvider::DicomObjSet &result)
{
  // Get all files in this dir
  // loop through them
  LOG( "Entering DIRECTORY: " << dirName);

  fs::directory_iterator end_iter;

  for ( fs::directory_iterator dir_itr( dirName );
        dir_itr != end_iter;
        ++dir_itr )
  {
    
    // if it is subdir, call itself on subdir
    if ( fs::is_directory( dir_itr->status() ) )
    {
      m_mainQueue.push( *dir_itr);
    }
    else
    {
      SolveFileGET( dir_itr->string(), patientID, studyID, serieID, result );
    }
  }
}

///////////////////////////////////////////////////////////////////////

void
LocalService::SolveFileGET( const std::string & fileName,
  const std::string &patientID,
	const std::string &studyID,
	const std::string &serieID,
  DcmProvider::DicomObjSet &result)
{
  OFString ofStr;

  DcmFileFormat dfile;
  OFCondition cond = dfile.loadFile( fileName.c_str());
  if (! cond.good())
  {
    LOG( "Loading of " << fileName << " failed. ("  << cond.text() << ")" );
    return;
  }

  DcmDataset *dataSet = dfile.getDataset();

  Entry entry;
  string setID;
  // get info about this file
  {
    dataSet->findAndGetOFString( DCM_PatientID, ofStr);
    entry.patID.append( ofStr.c_str() );

    dataSet->findAndGetOFString( DCM_StudyInstanceUID, ofStr);
    entry.studyID.append( ofStr.c_str() );

    dataSet->findAndGetOFString( DCM_SeriesInstanceUID, ofStr);
    setID.append( ofStr.c_str() );
  }

  // compare with wantend ones
  if( entry.patID == patientID
    && entry.studyID == studyID
    && setID == serieID)
  {
    // if it mathes, insert new image into result
    DicomObj buddy;
    result.push_back( buddy);

    // copy dataset reference & init
    DicomObj *newOne = &result.back();
    newOne->Load( fileName);
    newOne->Init();
  }
}

///////////////////////////////////////////////////////////////////////

void
LocalService::Reset(void)
{
  m_setOfEntries.clear();
}

///////////////////////////////////////////////////////////////////////

} // namespace
}

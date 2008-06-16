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
LocalService::GetImageSet(
      const std::string &patientID,
			const std::string &studyID,
			const std::string &serieID,
			Dicom::DcmProvider::DicomObjSet &result)
{
  
}

///////////////////////////////////////////////////////////////////////

void
LocalService::SolveDir( fs::path & dirName,
                       Dicom::DcmProvider::ResultSet &result)
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
  const std::string & fileName, const std::string & dirName,
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

  Entry entry;
  // get info about this file
  {
    dataSet->findAndGetOFString( DCM_PatientID, ofStr);
    entry.patID.append( ofStr.c_str() );

    dataSet->findAndGetOFString( DCM_StudyInstanceUID, ofStr);
    entry.studyID.append( ofStr.c_str() );

    //dataSet->findAndGetOFString( DCM_SeriesInstanceUID, ofStr);
    //entry.setID.append( ofStr.c_str() );
  }

  // look if it is already in result set
  SetOfEntries::iterator found = m_setOfEntries.find( entry);
  if( found == m_setOfEntries.end() )
  {
    // if not, put new line into result set
    Dicom::DcmProvider::TableRow row;
    GetTableRowFromDataSet( dataSet, &row);
    result.push_back( row);

    // put entry into set
    m_setOfEntries.insert( SetOfEntries::value_type( entry) );
  }
}

///////////////////////////////////////////////////////////////////////

} // namespace
}

/**
 *  @ingroup dicom
 *  @file LocalService.cpp
 *  @author Vaclav Klecanda
 */
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

///////////////////////////////////////////////////////////////////////////////

LocalService::LocalService()
{
  // tryes to open local db file (for case it don't exist)
  std::ofstream file(LOCAL_REC_DB_FILE_NAME);
  if( ! file.good() )
  {
    file.close();
    throw ExceptionBase("Could not create DB file!");
  }
  file.close();

  // open it for reading and load db
  std::ifstream forReadingFile( LOCAL_REC_DB_FILE_NAME);
  if( ! forReadingFile.good() )
  {
    forReadingFile.close();
    throw ExceptionBase("Cannot read the DB file!");
  }
  Load( forReadingFile);
  forReadingFile.close();
}

///////////////////////////////////////////////////////////////////////////////

LocalService::~LocalService()
{
  std::ofstream file(LOCAL_REC_DB_FILE_NAME);
  if( ! file.good() )
  {
    LOG("Could NOT write to DBFile");
  }
  else
  {
    Flush( file);
  }
  file.close();
}

///////////////////////////////////////////////////////////////////////////////

void
LocalService::Find( 
			Dicom::DcmProvider::ResultSet &result,
      const std::string &path)
{
  Reset();

  // save search dir
  //m_lastSearchDir = path;

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

LocalService::Series &
LocalService::GetSeries( const std::string &patientID,
			const std::string &studyID)
{
  // just take informatoin from tree
  Patients::iterator pat = m_patients.find( patientID);
  if( pat == m_patients.end() )
    throw ExceptionBase("Patient not found!");
  else
  {
    Studies &studies = pat->second.studies;
    Studies::iterator stud = studies.find( studyID);
    if( stud == studies.end() )
      throw ExceptionBase("Study not found!");
    else
      return stud->second.series;
  }
}

///////////////////////////////////////////////////////////////////////

void
LocalService::FindStudyInfo( 
      DcmProvider::SerieInfoVector &result,
      const std::string &patientID,
			const std::string &studyID)
{
  // just take informatoin from tree
  Series series = GetSeries( patientID, studyID);
  DcmProvider::SerieInfo s;
  for( Series::iterator serie=series.begin(); 
    serie != series.end(); serie++)
  {
    s.id = serie->second.id;
    s.description = serie->second.desc;
    result.push_back( s);
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
  // retrieve from tree, where the serie of image is
  Series &series = GetSeries( patientID, studyID);
  Series::iterator serie = series.find( serieID);

  // just for sure
  if( serie == series.end() )
    throw ExceptionBase("Not a serie found!");

  fs::path full_path = fs::system_complete( fs::path( 
    serie->second.path) );

  // and start to search from there
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
  const std::string & fileName, const std::string & path,
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

  DcmProvider::TableRow row;
  DcmProvider::SerieInfo serInfo;

  // load data and check if it is already in tree
  CheckDataSet( dataSet, serInfo, row, path);

  // look if this row is already in resultSet
  FoundStudiesSet::iterator found = m_alreadyFoundInRun.find(row.studyID);
  // if not, add it
  if( found == m_alreadyFoundInRun.end() )
  {
    m_alreadyFoundInRun.insert( FoundStudiesSet::value_type( row.studyID) );
    result.push_back( row);
  }
}

///////////////////////////////////////////////////////////////////////

void
LocalService::CheckDataSet(
    DcmDataset *dataSet,
    DcmProvider::SerieInfo &sInfo,
    DcmProvider::TableRow &row,
    std::string path)
{
  // load data from dataSet
  GetTableRowFromDataSet( dataSet, &row);
  GetSeriesInfo( dataSet, &sInfo);

  // now check what is in database
  Patients::iterator patIt = m_patients.find( row.patientID);
  if( patIt == m_patients.end() )
  {
    // insert new patient
    // insert new study
    Serie serie(sInfo.id, sInfo.description, path);
    Series s;
    s.insert( Series::value_type( serie.id, serie) );
    Study stud(row.studyID, row.date, s);
    Studies buddStudies;
    buddStudies.insert( Studies::value_type( stud.id, stud));

    Patient buddPat( row.patientID, row.name, row.birthDate, row.sex, buddStudies);
    m_patients.insert( Patients::value_type( row.patientID, buddPat) );
  }
  else
  {
    // perform lookup level down
    Studies &studies = patIt->second.studies;
    Studies::iterator studItr = studies.find( row.studyID);
    if( studItr == studies.end() )
    {
      // insert new study
      Serie serie(sInfo.id, sInfo.description, path);
      Series s;
      s.insert( Series::value_type( serie.id, serie) );
      Study stud(row.studyID, row.date, s);
      patIt->second.studies.insert( Studies::value_type(
        stud.id, stud) );
    }
    else
    {
      // perform lookup level down
      Series &series = studItr->second.series;
      Series::iterator serItr = series.find( sInfo.id);
      if( serItr == series.end() )
      {
        // insert new serie
        Serie buddy(sInfo.id, sInfo.description, path);
        series.insert( Series::value_type( sInfo.id, buddy) );
      }
      // else do nothing
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

// flushing strings
static void WriteString ( std::ofstream &s, std::string &val)
{
  s << (uint32) val.size() << " ";
  for( uint32 i=0; i< val.size(); i++)
  {
    s.put(val[i]);
  }
  s << " ";
}

///////////////////////////////////////////////////////////////////////////////

void
LocalService::Flush( std::ofstream &stream)
{
  // write size of patients
  stream << (uint32) m_patients.size() << " ";

  // go through the tree and flush it
  for( Patients::iterator patIt = m_patients.begin(); 
    patIt != m_patients.end(); patIt++)
  {
    WriteString( stream, patIt->second.id);
    WriteString( stream, patIt->second.name);
    WriteString( stream, patIt->second.bornDate);
    stream << (uint8) patIt->second.sex << " ";

    // write size of studies
    stream << (uint32) patIt->second.studies.size() << " ";

    for( Studies::iterator studItr = patIt->second.studies.begin();
      studItr != patIt->second.studies.end(); studItr++)
    {
      WriteString( stream, studItr->second.id);
      WriteString( stream, studItr->second.date);

      // write size of series
      stream << (uint32) studItr->second.series.size() << " ";

      for( Series::iterator serItr = studItr->second.series.begin();
        serItr != studItr->second.series.end(); serItr++)
      {
        WriteString( stream, serItr->second.id);
        WriteString( stream, serItr->second.desc);
        WriteString( stream, serItr->second.path);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

// loading strings
static void LoadString ( std::ifstream &s, std::string &val)
{
  uint32 size;
  int8 tmp;

  s >> size;
  tmp = (uint8) s.get();  // skip space

  for( uint32 i=0; i<size; i++)
  {
    tmp = (uint8) s.get();
    val.append( (const char *) &tmp, 1);
  }
}

///////////////////////////////////////////////////////////////////////////////

void
LocalService::Load( std::ifstream &stream)
{
  uint32 patCount;
  uint32 studyCount;
  uint32 seriesCount;

  stream >> patCount;

  if(stream.eof() )
    return;

  // load patients
  for( uint32 pats=0; pats < patCount; pats++)
  {
    Patient pat;
    LoadString( stream, pat.id);
    LoadString( stream, pat.name);
    LoadString( stream, pat.bornDate);
    stream >> pat.sex;

    stream >> studyCount;
    for( uint32 studs=0; studs < studyCount; studs++)
    {
      Study study;

      LoadString( stream, study.id);
      LoadString( stream, study.date);

      // load series
      stream >> seriesCount;
      for( uint32 sers=0; sers < seriesCount; sers++)
      {
        Serie ser;
        LoadString( stream, ser.id);
        LoadString( stream, ser.desc);
        LoadString( stream, ser.path);
        study.series.insert( Series::value_type(ser.id, ser) );
      }

      pat.studies.insert( Studies::value_type( study.id, study) );
    }

    m_patients.insert( Patients::value_type( pat.id, pat) );
  }
}

///////////////////////////////////////////////////////////////////////////////

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
      SolveFileGET( dir_itr->string(), 
        patientID, studyID, serieID, result, dirName.string() );
    }
  }
}

///////////////////////////////////////////////////////////////////////

void
LocalService::SolveFileGET( const std::string & fileName,
  const std::string &patientID,
	const std::string &studyID,
	const std::string &serieID,
  DcmProvider::DicomObjSet &result,
  const std::string &path)
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

  DcmProvider::TableRow row;
  DcmProvider::SerieInfo serInfo;

  // load data and check if it is already in tree
  CheckDataSet( dataSet, serInfo, row, path);

  // compare with wantend ones
  if( row.patientID == patientID
    && row.studyID == studyID
    && serInfo.id == serieID)
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
  m_alreadyFoundInRun.clear();
}

///////////////////////////////////////////////////////////////////////

} // namespace
}
/** @} */


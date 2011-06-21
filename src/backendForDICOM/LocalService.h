#ifndef M4DDICOMLOCALSERVICE
#define M4DDICOMLOCALSERVICE

/**
 *  @ingroup dicom
 *  @file LocalService.h
 *  @author Vaclav Klecanda
 *  @{
 */

#include <string>
#include <map>
#include <set>
#include "structures.h"
#include "DcmObject.h"
#include "common/ProgressNotifier.h"

namespace M4D
{
namespace Dicom
{

#define LOCAL_REC_DB_FILE_NAME "tree.dat"

/// Implements searching and gettting functions to local FS dicom files.
/**
 *  It sequentialy loads data files in specified folder (and subfolders through queue), read ID info, based on that info and given filter inserts or not inserts (if matching found) record into result.
 *  Each search run writes the folder that is performed on, build structure of information that is used when aditional informations concerning data from the same run are required. 
 *  One run is quite expensive while loading each file is needed (there is no other way how to read required IDs). So it is sensitive how wide and deep is the subtree of the given folder.
 *  Maybe some timeouts will be required.
 *  All functions are private beacause are all called from friend class DcmProvider.
 */
class LocalService
{
public:
	// performs search run on given folder
	void 
	Find( 
		ResultSet &result,
		const std::string &path
		);

	// returns serie info based on build info structure
	void 
	FindStudyInfo( 
		SerieInfoVector &result,
		const std::string &patientID,
		const std::string &studyID
		);

	// performs search run and returns set of loaded data files (DicomObj)
	void 
	GetImageSet(
		const std::string &patientID,
		const std::string &studyID,
		const std::string &serieID,
		DicomObjSet &result
		);

	LocalService();
	~LocalService();

	void 
	GetSeriesFromFolder( 
		std::string aFolder,
		std::string aPatientID,
		std::string aStudyID,
		std::string aSerieID,
		DicomObjSet &aResult,
		ProgressNotifier::Ptr aProgressNotifier = ProgressNotifier::Ptr()
		);

private:
	struct Serie
	{
		std::string id;
		std::string desc;
		std::string path;

		Serie() {}

		Serie( std::string id_, std::string desc_, std::string path_)
			: id(id_), desc( desc_), path( path_)
		{
		}

		Serie( const Serie &other)
			: id(other.id), desc( other.desc), path( other.path)
		{
		}

		inline bool operator< (const Serie &b) const
		{
			return id.compare( b.id) < 0;
		}
	};
	typedef std::map<std::string, Serie> Series;

	struct Study
	{
		std::string id;
		std::string date;
		Series series;

		Study() {}

		Study( std::string id_, std::string date_, Series series_)
			: id(id_), date( date_), series( series_)
		{
		}

		Study( const Study &other)
			: id(other.id), date( other.date), series( other.series)
		{
		}

		inline bool operator< (const Study &b) const
		{
			return id.compare( b.id) < 0;
		}
	};
	typedef std::map<std::string, Study> Studies;

	struct Patient
	{
		std::string id;
		std::string name;
		std::string bornDate;
		uint8 sex;
		Studies studies;

		Patient() {}

		Patient( 
			std::string id_, 
			std::string name_,
			std::string bornDate_, 
			bool sex_, 
			Studies studies_
			) : id(id_), name( name_), bornDate( bornDate_), sex( sex_), studies( studies_)
		{
		}

		Patient( const Patient &other)
		: id(other.id), name( other.name), bornDate( other.bornDate), sex( other.sex), studies( other.studies)
		{
		}

		inline bool operator< (const Patient &b) const
		{
			return id.compare( b.id) < 0;
		}
	};
	typedef std::map<std::string, Patient> Patients;

	Patients m_patients;

	typedef std::set<std::string> FoundStudiesSet;

	FoundStudiesSet m_alreadyFoundInRun;

	void Reset(void);

	// queue of remainig subfolders
	std::queue<boost::filesystem::path> m_mainQueue;

	// currently searched folder
	std::string m_lastSearchDir;

	ResultSet m_lastResultSet;

	// supporting functions to go on one folder or to solve single file
	void SolveDir( boost::filesystem::path & dirName,
	ResultSet &result);
	// ...
	void SolveFile( const boost::filesystem::path & fileName,
	const boost::filesystem::path & path,
	ResultSet &result);
	// ...
	void SolveDirGET( boost::filesystem::path & dirName,
	const std::string &patientID,
	const std::string &studyID,
	const std::string &serieID,
	DicomObjSet &result);
	// ...
	void SolveFileGET( const boost::filesystem::path & fileName,
	const std::string &patientID,
	const std::string &studyID,
	const std::string &serieID,
	DicomObjSet &result,
	const std::string &path);

	void CheckDataSet( 
	DcmDataset *dataSet,
	SerieInfo &sInfo,
	TableRow &row,
	std::string path);

	Series &GetSeries( const std::string &patientID,
	const std::string &studyID);

	void Flush( std::ofstream &stream);
	void Load( std::ifstream &stream);
};

} // namespace
}

/** @} */

#endif


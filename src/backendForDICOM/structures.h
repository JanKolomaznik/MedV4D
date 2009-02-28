#ifndef STRUCTURES_H_
#define STRUCTURES_H_

namespace M4D {
namespace Dicom {

/// Represents one row in table that shows found results.
struct TableRow {
	std::string patientID;
	std::string name;
	std::string birthDate;
	bool sex;
	//std::string accesion;
	std::string studyID;
	std::string date;
	std::string time;
	std::string modality;
	std::string description;
	std::string referringMD;
	//std::string institution;
	//std::string location;
	//std::string server;
	//std::string availability;
	//std::string status;
	//std::string user;
};

/// Result set - Vector of table rows.
typedef std::vector<TableRow> ResultSet;

/// Contains all series' infos of one Study.
struct SerieInfo {
	std::string id;
	std::string description;

	bool operator <(const SerieInfo &b) const {
		return (id + description).compare(b.id + b.description) < 0;
	}
};

/// Vector of SerieInfos
typedef std::vector<SerieInfo> SerieInfoVector;

// vector of M4DSetInfo
typedef std::vector<std::string> StringVector;
typedef std::map<std::string, StringVector> StudyInfo;

/// Container for one serie of images
typedef std::vector<DicomObj> DicomObjSet;
/// shared pointer to DicomObjSet type
typedef boost::shared_ptr< DicomObjSet> DicomObjSetPtr;

}
}
#endif /*STRUCTURES_H_*/

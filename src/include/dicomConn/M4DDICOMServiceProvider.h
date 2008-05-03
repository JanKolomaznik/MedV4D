#ifndef M4D_DICOM_SERVICE_PROVIDER
#define M4D_DICOM_SERVICE_PROVIDER

#include <vector>
#include <map>
#include <string>
#include <boost/shared_ptr.hpp>

namespace M4D
{
namespace Dicom {

	class DcmProvider
	{
	private:
		void *findService;
		void *moveService;
	public:

		// TYPEDEFS ///////////////////////////////////////////////////////////

		class DicomObj;

		// represents one row in table that shows found results
		typedef struct s_browserRow {
			std::string patientName;
			bool patientSex;
			std::string patientBirthDate;
			std::string patentID;
			std::string studyID;
			std::string studyDate;
			std::string modality;
		} TableRow;

		typedef std::vector<TableRow> ResultSet;

		// vector of image IDs
		typedef std::vector<std::string> StringVector;
		// vector of M4DSetInfo
		typedef std::map<std::string, StringVector> StudyInfo;

		typedef std::vector<DicomObj> DicomObjSet;
		typedef boost::shared_ptr< DicomObjSet > DicomObjSetPtr;

		// METHODs ////////////////////////////////////////////////////////////

		// user inputs some values into seach form & wants a table to be filled
		// with appropriate results.
		/**
		 *	@param result - reference to result set object
		 *	@param patientName - reference to string containing ...
		 *	@param patientID   -   - || -
		 *	@param modalities  - reference to string containing set 
		 *		of wanted modalities divided by '\' character
		 *	@param date		   - ref. to string containing one of theese variants:
		 *		<dateFrom>- = all items that has date >= than 'dateFrom'
		 *		-<dateTo>	= all items thats' date <= than 'dateTo'
		 *		<dateFrom>-<dateTo> = all items between that dates
		 *		<dateX> = string with format YYYYMMDD
		 */
		void Find( 
			DcmProvider::ResultSet &result,
			const std::string &patientName,
			const std::string &patientID,
			const StringVector &modalities,
			const std::string &dateFrom,
			const std::string &dateTo) ;

		// user select any study and wants to get seriesInstanceUIDs
		void FindStudyInfo(
			const std::string &patientID,
			const std::string &studyID,
			StringVector &info) ;

		// the same as FindStudyInfo but gets even imageIDs
		void WholeFindStudyInfo(
			const std::string &patientID,
			const std::string &studyID,
			StudyInfo &info) ;

		// finds all studies concerning given patient
		void FindStudiesAboutPatient(  
			const std::string &patientID,
			ResultSet &result) ;

		void GetImage(
			const std::string &patientID,
			const std::string &studyID,
			const std::string &serieID,
			const std::string &imageID,
			DicomObj &object) ;

		void GetImageSet(
			const std::string &patientID,
			const std::string &studyID,
			const std::string &serieID,
			DicomObjSet &result) ;

		DcmProvider();
		~DcmProvider();
	};

};
}
#include "M4DDICOMObject.h"

#endif

#ifndef M4DDICOMFINDSERVICE
#define M4DDICOMFINDSERVICE

/**
 *  Implements C-FIND service to DICOM server
 */

using namespace std;

class M4DFindService
{
private :
	DcmDataset *queryForFilterTable;

	void GetQueryForFilterTable( void);

	M4DDicomAssociation *m_assocToServer;
	T_ASC_Network *net;

	void GetSeriesQuery( DcmDataset **query);
	void GetImageQuery( 
		DcmDataset **query,
		std::string studyID,
		std::string serieID);

	static void
	ProgressCallback(
        void *callbackData,
        T_DIMSE_C_FindRQ *request,
        int responseCount,
        T_DIMSE_C_FindRSP *rsp,
        DcmDataset *responseIdentifiers );

public:
	M4DFindService();
	~M4DFindService();

	void Find( M4DDicomServiceProvider::ResultSet &result) throw (...);
	
	
};

#endif
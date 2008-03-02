#ifndef M4DDICOMASSOC
#define M4DDICOMASSOC

#include <map>
#include <fstream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////

class M4DDicomAssociation
{
private:

	/** 
	* represents all neccessary informations for DICOM
	* network association
	*/
	typedef struct s_DICOMAddress {
		string callingAPTitle;
		string calledDBName;
		string calledServerName;
		int calledServerPort;
		string transferModel;
		E_TransferSyntax transferSyntax;
	} DICOMAddress;

	typedef map<string, DICOMAddress *> AddressContainer;

	//-----------------------------------

	T_ASC_Association *m_assoc;
	DICOMAddress *m_assocAddr;
	T_ASC_Parameters *m_assocParams;

	void AddPresentationContext(T_ASC_Parameters *params);
	
	// addresses stuff
	static AddressContainer addressContainer;

	static DICOMAddress *GetAddress( string addressPointer) throw (...);
	static void FindNonCommentLine( ifstream &f, string &line);
	static void LoadOneAddress( ifstream &f) throw (...);
	static void InitAddressContainer( void) throw (...);	// loads the config.cfg

public:
	M4DDicomAssociation( string assocID);
	~M4DDicomAssociation( void);

	void Release( void);
	void Abort( void);
	void Request( T_ASC_Network *net) throw (...);
	
	inline T_ASC_Association *GetAssociation( void)	{ return m_assoc; }
	inline DICOMAddress *GetAssocAddress( void) { return m_assocAddr; }
};

#endif
#ifndef M4DDICOMASSOC_H
#define M4DDICOMASSOC_H

#include <map>
#include <fstream>

namespace M4D
{
namespace DicomInternal 
{
///////////////////////////////////////////////////////////////////////////////

class M4DDicomAssociation
{
private:

 /** 
	* represents all neccessary informations for DICOM
	* network association establishing
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

  // supports for loading config.cfg file that contain addresses
  // to server connections
	static DICOMAddress *GetAddress( string addressPointer) ;
	static void FindNonCommentLine( ifstream &f, string &line);
	static void LoadOneAddress( ifstream &f) ;
	static void InitAddressContainer( void) ;	// loads the config.cfg

public:
  // ctor, dtor
	M4DDicomAssociation( string assocID);
	~M4DDicomAssociation( void);

	void Release( void);
	void Abort( void);
	void Request( T_ASC_Network *net) ;
	
	inline T_ASC_Association *GetAssociation( void)	{ return m_assoc; }
	inline DICOMAddress *GetAssocAddress( void) { return m_assocAddr; }

	T_ASC_PresentationContextID FindPresentationCtx( void) ;
};

} // namespace
}
#endif


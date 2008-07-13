#ifndef M4DDICOMASSOC_H
#define M4DDICOMASSOC_H

#include <map>
#include <fstream>

namespace M4D
{
namespace DicomInternal 
{
///////////////////////////////////////////////////////////////////////////////

/**
 *  This is base class for DICOM assotiation. Assotiation is something like connection. It is defined by IP address, port and 'Application Entity' (AE). AE is like name. Both sides (client and server) has its own AE.
 *  This class contains pointers to DCMTK library objects that contains actual assotiation and his properties. As well as some action members that establish (request), aborts and terminate the assotiation.
 *  Next item contained in this class is address container that holds neccessary properties for different assotiation. Assotiation has different properties when different services are called. So the container is a map indexed by string eith called service. The container is filled from config file. There are some supporting methodes taking care of it. The container is shared between all instances (static).
 */
class M4DDicomAssociation
{
private:

 /** 
	* Represents all neccessary informations for DICOM
	* network association establishing.
	*/
	struct DICOMAddress 
  {
		string callingAPTitle;    /// our AE Title
		string calledDBName;      /// DB whose data we want to retrieve 
		string calledServerName;  /// DNS name (or IP)
		int calledServerPort;     /// port where server runs
		string transferModel;     /// transfer model of transmission
    /// trasfer syntax (data format) little/big endian, JPEG formatted, ...
		E_TransferSyntax transferSyntax;
	};

	typedef map<string, DICOMAddress *> AddressContainer;

	//-----------------------------------

  // pointer to DCMTK objects
	T_ASC_Association *m_assoc;
	DICOMAddress *m_assocAddr;
	T_ASC_Parameters *m_assocParams;

	void AddPresentationContext(T_ASC_Parameters *params);
	
	// addresses stuff. Shared between all instances
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


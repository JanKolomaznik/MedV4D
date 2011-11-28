#ifndef M4DDICOMASSOC_H
#define M4DDICOMASSOC_H

#include <map>
#include <fstream>
#include <string>

/**
 *  @ingroup dicom
 *  @file DicomAssoc.h
 *  @author Vaclav Klecanda
 *  @{
 */

namespace M4D
{
namespace Dicom 
{

/**
*  This is base class for DICOM assotiation. Assotiation is something like connection. It is defined by IP address, port and 'Application Entity' (AE). AE is like name. Both sides (client and server) has its own AE.
*  This class contains pointers to DCMTK library objects that contains actual assotiation and his properties. As well as some action members that establish (request), aborts and terminate the assotiation.
*  Next item contained in this class is address container that holds neccessary properties for different assotiation. Assotiation has different properties when different services are called. So the container is a map indexed by string eith called service. The container is filled from config file. There are some supporting methodes taking care of it. The container is shared between all instances (static).
*/
class DicomAssociation
{
private:

	/** 
	* Represents all neccessary informations for DICOM
	* network association establishing.
	*/
	struct DICOMAddress 
	{
		std::string callingAPTitle;    //< our AE Title
		std::string calledDBName;      //< DB whose data we want to retrieve 
		std::string calledServerName;  //< DNS name (or IP)
		int calledServerPort;     //< port where server runs
		std::string transferModel;     //< transfer model of transmission
		E_TransferSyntax transferSyntax;//< trasfer syntax (data format) little/big endian, JPEG formatted, ...
	};

	typedef std::map<std::string, DICOMAddress *> AddressContainer;

	void 
	AddPresentationContext(T_ASC_Parameters *params);

	// supports for loading config.cfg file that contain addresses
	// to server connections
	static DICOMAddress *
	GetAddress( std::string addressPointer) ;
	
	static void 
	FindNonCommentLine( std::ifstream &f, std::string &line);
	
	static void 
	LoadOneAddress( std::ifstream &f) ;
	//-----------------------------------

	// pointer to DCMTK objects
	T_ASC_Association *m_assoc;
	DICOMAddress *m_assocAddr;
	T_ASC_Parameters *m_assocParams;

	// addresses stuff. Shared between all instances
	static AddressContainer addressContainer;


	static bool _configFilePresent;

public:
	// ctor, dtor
	DicomAssociation( std::string assocID );
	~DicomAssociation( void );

	static bool 
	InitAddressContainer( void );	//< loads the config.cfg

	void 
	Release( void );
	void 
	Abort( void );
	void 
	Request( T_ASC_Network *net );

	inline T_ASC_Association *
	GetAssociation( void )
	{ 
		return m_assoc; 
	}

	inline DICOMAddress *
	GetAssocAddress( void ) 
	{ 
		return m_assocAddr; 
	}

	T_ASC_PresentationContextID 
	FindPresentationCtx( void );
};

} // namespace
}
/** @} */
#endif


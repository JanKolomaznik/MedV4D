#ifndef M4DDICOM_ABSTRACT_SERVICE
#define M4DDICOM_ABSTRACT_SERVICE

/**
 *  This is base class for all services that is requested to the side of DICOM server. There is poiner to DCMTK Network object which need network subsystem on windows system initialized at the beginning of usage and unloaded when is no more needed. So there is reference counting.
 *  Each service is devided into 2 parts. SCP (Service Class Producer = server) and SCU (Sevice Class User = Client). Both sides of an service has to agree while establishing assotiation what role to play. Normally DICOM server plays server role (SCP) for C-FIND, C-MOVE, C-STORE but for C-MOVE subassotiations when image are transfered to client plays SCU role (it requests sending of data). Each scenario is described in doc od successors.
 *  Another class member is query dataSet that is used as a query to server, similary like SQL query string. Each query can be done on one of 4 levels: Patient, Study, Series, Images. For example: for Study level are all matched studies returned, for Series Level all matched series, ... On each level are relevant only sepecified set of matchable attributes so its quite hard to send robust query. Some other filtering has sometimes to be done on returned records.
 *  Common scenario of all services is to prepare query dataSet that selects wanted data files. Then proceed the query to main TCMTK performing function and then retrieve resulting data through callbacks to final data structures. Ancesting classes implementing the services contain supporting callback definitions that cooperated with DCMTK functions and definitions of structures that are then passed to appropriate callbacks.
 *  Code of services is inspired by pilot implementation of DCMTK utilities.
 */
namespace M4D
{
namespace DicomInternal 
{

class AbstractService
{
protected :
	static size_t m_numOfInstances;
	static const size_t m_maxPDU;

	M4DDicomAssociation *m_assocToServer;
	T_ASC_Network *m_net;
	
	DcmDataset *m_query;

protected:
	AbstractService();
	~AbstractService();	
	
};

} // namespace
}/*namespace M4D*/

#endif


#ifndef M4DDICOM_ABSTRACT_SERVICE
#define M4DDICOM_ABSTRACT_SERVICE

/**
 *  Implements C-FIND service to DICOM server
 */

using namespace std;

class M4DAbstractService
{
protected :
	static size_t m_numOfInstances;
	static const size_t m_maxPDU;

	M4DDicomAssociation *m_assocToServer;
	T_ASC_Network *m_net;
	
	DcmDataset *m_query;

protected:
	M4DAbstractService();
	~M4DAbstractService();	
	
};

#endif
#include "dcmtk/dcmnet/dimse.h"
#include "dcmtk/dcmnet/diutil.h"
#include "dcmtk/dcmdata/dcdeftag.h"

#include "M4DDicomAssoc.h"
#include "M4DDICOMServiceProvider.h"
#include "AbstractService.h"
#include "main.h"

///////////////////////////////////////////////////////////////////////

// static members definitions
size_t 
M4DAbstractService::m_numOfInstances;

const size_t
M4DAbstractService::m_maxPDU = 16384;

///////////////////////////////////////////////////////////////////////

M4DAbstractService::M4DAbstractService()
{
	// if it is the first instance let's init network
	if( m_numOfInstances == 0)
	{
		/*
		** Don't let dcmdata remove tailing blank padding or perform other
		** maipulations.  We want to see the real data.
		*/
		dcmEnableAutomaticInputDataCorrection.set(OFFalse);

		/* make sure data dictionary is loaded */
		/*if (!dcmDataDict.isDictionaryLoaded()) {
			fprintf(stderr, "Warning: no data dictionary loaded, check environment variable: %s\n",
					DCM_DICT_ENVIRONMENT_VARIABLE);
		}*/

	#ifdef HAVE_WINSOCK_H
		WSAData winSockData;
		/* we need at least version 1.1 */
		WORD winSockVersionNeeded = MAKEWORD( 1, 1 );
		WSAStartup(winSockVersionNeeded, &winSockData);
	#endif
	}

	m_numOfInstances++;
}

///////////////////////////////////////////////////////////////////////

M4DAbstractService::~M4DAbstractService()
{
	// if it is the last instance let's clean up network
	if( m_numOfInstances == 1)
	{
	#ifdef HAVE_WINSOCK_H
		WSACleanup();
	#endif		
	}

	m_numOfInstances--;

	/* drop the network, i.e. free memory of T_ASC_Network* structure. This call */
	/* is the counterpart of ASC_initializeNetwork(...) which was called above. */
	if( ASC_dropNetwork(&m_net).bad() )
	{
		D_PRINT( "Drop network failed!");
	}

}

///////////////////////////////////////////////////////////////////////
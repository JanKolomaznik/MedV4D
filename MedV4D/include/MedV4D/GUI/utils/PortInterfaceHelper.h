#ifndef PORT_INTERFACE_HELPER_H
#define PORT_INTERFACE_HELPER_H

#include <boost/mpl/vector.hpp>
#include <boost/mpl/transform.hpp>
#include "MedV4D/Imaging/Ports.h"
#include "MedV4D/Common/Functors.h"
#include "MedV4D/Common/Common.h"
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/size.hpp>

#ifndef Q_MOC_RUN
#include <boost/fusion/container/vector/convert.hpp>
#include <boost/fusion/include/as_vector.hpp>
#endif



namespace M4D
{

/**
 * \tparam DatasetTypeVector mpl::vector of dataset types.
 **/
template< typename DatasetTypeVector >
class PortInterfaceHelper: public M4D::Imaging::MessageReceiverInterface
{
public:
	static const size_t PortCount = boost::mpl::size< DatasetTypeVector >::value;
	template< int ID >
	struct Dataset { 
		typedef boost::mpl::at< DatasetTypeVector, boost::mpl::int_< ID > > type;
	};

	PortInterfaceHelper();

	/**
	 * @return Returns list of all available input ports.
	 **/
	const Imaging::InputPortList &
	InputPort()const
		{ return mInputPorts; }

	/**
	 * It tries to fetch dataset from each port in port list and lock it for reading. 
	 * If something fails apropriate exception is thrown.
	 **/
	void
	TryGetAndLockAllInputs();

	/**
	 * Same as TryGetAndLockAllInputs(), but check only connected ports
	 **/

	size_t
	TryGetAndLockAllAvailableInputs();
	/**
	 * Releases all locked datasets and resets pointers
	 **/
	void
	ReleaseAllInputs();

	void
	ReceiveMessage( 
		M4D::Imaging::PipelineMessage::Ptr 			msg, 
		M4D::Imaging::PipelineMessage::MessageSendStyle 	sendStyle, 
		M4D::Imaging::FlowDirection				direction
		);

protected:
	//typedef typename boost::fusion::result_of::as_vector<DatasetTypeVector>::type InputDatasetList;

	//InputDatasetList			mInputDatasets;
	std::vector< M4D::Imaging::ADataset::ConstWPtr >	mInputDatasets;//[boost::mpl::size< DatasetTypeVector >::value];
	std::vector< M4D::Common::TimeStamp >			mTimeStamps;//[boost::mpl::size< DatasetTypeVector >::value];
private:
	struct PortCreator
	{
		PortCreator( M4D::Imaging::InputPortList &portList ): mInputPorts( portList ) {}

		template< typename DatasetTypeWrapper > 
		void 
		operator()( const DatasetTypeWrapper &x)
		{
			typedef M4D::Imaging::InputPortTyped< typename DatasetTypeWrapper::type > PortType;

			PortType *ptr = new PortType();
			mInputPorts.AppendPort( ptr );
		}
		M4D::Imaging::InputPortList		&mInputPorts;
	};

	

	M4D::Imaging::InputPortList		mInputPorts;
};




template< typename DatasetTypeVector >
PortInterfaceHelper< DatasetTypeVector >
::PortInterfaceHelper() : mInputDatasets( PortInterfaceHelper< DatasetTypeVector >::PortCount ), mTimeStamps( PortInterfaceHelper< DatasetTypeVector >::PortCount ), mInputPorts( this )
{
	typedef typename boost::mpl::transform< DatasetTypeVector, M4D::Functors::MakeTypeBox >::type wrapedTypes;

	boost::mpl::for_each< wrapedTypes >( PortCreator( mInputPorts ) );
}

template< typename DatasetTypeVector >
void
PortInterfaceHelper< DatasetTypeVector >
::TryGetAndLockAllInputs()
{
	//TODO
	mInputDatasets[0] = mInputPorts.GetPort(0).GetDatasetPtr();
	mInputDatasets[1] = mInputPorts.GetPort(1).GetDatasetPtr();
	//_THROW_ ErrorHandling::ETODO();
}

template< typename DatasetTypeVector >
size_t
PortInterfaceHelper< DatasetTypeVector >
::TryGetAndLockAllAvailableInputs()
{
	//TODO
	size_t locked_count = 0;
	for( size_t i = 0; i < PortCount; ++i ) {
		mInputDatasets[i] = M4D::Imaging::ADataset::ConstPtr();
	}
	for( size_t i = 0; i < PortCount; ++i ) {
		M4D::Imaging::InputPort& port = mInputPorts.GetPort(i);
		if( port.IsPlugged() && port.IsDatasetAvailable() ) {
			try {
				mInputDatasets[i] = port.GetDatasetPtr();
				++locked_count;
			} catch (...) {
				D_PRINT( "Dataset not available on port n." << i );
			}
		}
	}
	
	return locked_count;
	/*if( mInputPorts.GetPort(1).IsPlugged() ) {
		mInputDatasets[1] = mInputPorts.GetPort(1).GetDatasetPtr();
	}*/
	
	//bool result = mInputPorts.GetPort(0).TryLockDataset();
	//_THROW_ ErrorHandling::ETODO();
}

template< typename DatasetTypeVector >
void
PortInterfaceHelper< DatasetTypeVector >
::ReleaseAllInputs()
{
	//TODO
	//_THROW_ ErrorHandling::ETODO();
}

template< typename DatasetTypeVector >
void
PortInterfaceHelper< DatasetTypeVector >
::ReceiveMessage( 
	M4D::Imaging::PipelineMessage::Ptr 			msg, 
	M4D::Imaging::PipelineMessage::MessageSendStyle 	sendStyle, 
	M4D::Imaging::FlowDirection				direction
	)
{

}

}/*namespace M4D*/


#endif /*PORT_INTERFACE_HELPER_H*/

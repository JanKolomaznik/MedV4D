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
#include <boost/fusion/container/vector/convert.hpp>
#include <boost/fusion/include/as_vector.hpp>




namespace M4D
{

/**
 * \tparam DatasetTypeVector mpl::vector of dataset types.
 **/
template< typename DatasetTypeVector >
class PortInterfaceHelper: public M4D::Imaging::MessageReceiverInterface
{
public:
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

	void
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
	M4D::Imaging::ADataset::ConstPtr		mInputDatasets[boost::mpl::size< DatasetTypeVector >::value];
	M4D::Common::TimeStamp				mTimeStamps[boost::mpl::size< DatasetTypeVector >::value];
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
::PortInterfaceHelper() : mInputPorts( this )
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
	//_THROW_ ErrorHandling::ETODO();
}

template< typename DatasetTypeVector >
void
PortInterfaceHelper< DatasetTypeVector >
::TryGetAndLockAllAvailableInputs()
{
	//TODO
	mInputDatasets[0] = mInputPorts.GetPort(0).GetDatasetPtr();
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

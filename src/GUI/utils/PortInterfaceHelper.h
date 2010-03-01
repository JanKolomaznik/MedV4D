#ifndef PORT_INTERFACE_HELPER_H
#define PORT_INTERFACE_HELPER_H

#include <boost/mpl/vector.hpp>
#include <boost/mpl/transform.hpp>
#include "Imaging/Ports.h"
#include "common/Functors.h"
#include "common/Common.h"
#include <boost/mpl/for_each.hpp>

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
		typedef boost::mpl::at< DatasetTypeVector, ID > type
	}

	PortInterfaceHelper();

	/**
	 * @return Returns list of all available input ports.
	 **/
	const Imaging::InputPortList &
	InputPort()const
		{ return _inputPorts; }

	/**
	 * It tries to fetch dataset from each port in port list and lock it for reading. 
	 * If something fails apropriate exception is thrown.
	 **/
	void
	TryGetAndLockAllInputs();

	/**
	 * Releases all locked datasets and resets pointers
	 **/
	void
	ReleaseAllInputs();
protected:

private:
	struct PortCreator
	{
		PortCreator( M4D::Imaging::InputPortList &portList ): _inputPorts( portList ) {}

		template< typename DatasetTypeWrapper > 
		void 
		operator()( const DatasetTypeWrapper &x)
		{
			typedef InputPortTyped< DatasetTypeWrapper::type > PortType;

			PortType *ptr = new PortType();
			_inputPorts.AppendPort( ptr );
		}
		M4D::Imaging::InputPortList		&_inputPorts;
	};

	

	M4D::Imaging::InputPortList		_inputPorts;
};




template< typename DatasetTypeVector >
PortInterfaceHelper< DatasetTypeVector >
::PortInterfaceHelper() : _inputPorts( *this )
{
	typedef transform< DatasetTypeVector, M4D::Functors::MakeTypeBox >::type wrapedTypes;

	boost::mpl::for_each< wrapedTypes >( PortCreator( _inputPorts ) );
}

template< typename DatasetTypeVector >
void
PortInterfaceHelper< DatasetTypeVector >
::TryGetAndLockAllInputs()
{
	//TODO
	_THROW_ ErrorHandling::ETODO();
}

template< typename DatasetTypeVector >
void
PortInterfaceHelper< DatasetTypeVector >
::ReleaseAllInputs()
{
	//TODO
	_THROW_ ErrorHandling::ETODO();
}


}/*namespace M4D*/


#endif /*PORT_INTERFACE_HELPER_H*/

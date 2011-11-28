#include "MedV4D/Common/IDGenerator.h"

namespace M4D
{
namespace Common
{

IDGenerator::IDGenerator( IDNumber initialID ): _lastID( initialID )
{

}

IDGenerator::~IDGenerator()
{

}

IDNumber
IDGenerator::NewID()
{
	Multithreading::ScopedLock lock( _accessLock );

	return ++_lastID;
}


}/*namespace Common*/
}/*namespace M4D*/


#ifndef ID_GENERATOR
#define ID_GENERATOR

#include "Common.h"
#include "Thread.h"


namespace M4D
{
namespace Common
{

typedef uint32	IDNumber;


class IDGenerator
{
public:
	IDGenerator( IDNumber initialID = 0 );

	~IDGenerator();

	IDNumber
	NewID();
private:

	IDNumber	_lastID;

	Multithreading::Mutex	_accessLock;
};

}/*namespace Common*/
}/*namespace M4D*/

/** @} */

#endif /*ID_GENERATOR*/



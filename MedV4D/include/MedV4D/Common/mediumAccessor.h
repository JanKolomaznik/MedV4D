#ifndef MEDIUMACCESSOR_H_
#define MEDIUMACCESSOR_H_

#include <memory>
#include <boost/make_shared.hpp>

namespace M4D
{
namespace IO
{

class MediumAccessor
{
public:
	typedef std::shared_ptr< MediumAccessor > Ptr;
	
	virtual ~MediumAccessor() {}
	
	virtual void PutData(const void *data, size_t length) = 0;
	virtual size_t GetData(void *data, size_t length) = 0;

	virtual bool eof() = 0;
};

}
}

#endif /*MEDIUMACCESSOR_H_*/

#ifndef ADDRESS_H_
#define ADDRESS_H_

namespace M4D
{
namespace Cell
{

#define Address2Pointer(add) ((void *)add.Get64())

class Address
{
public:
	Address() { long_addr.ull = 0; }
//	Address(unsigned long long add) { long_addr.ull = add; }
	Address(const void *p) 
	{ 
		//if(sizeof(p) == 8)
			long_addr.ull = (unsigned long long) p;
//		else
//		{
//			long_addr.ui[0] = 0;
//			long_addr.ui[1] = (unsigned int) p;
//		}
	}
	unsigned long long Get64() const { return long_addr.ull; }
	uint32 GetLo() { return long_addr.ui[1]; }
	uint32 GetHi() { return long_addr.ui[0]; }
	
	inline bool operator !=(const Address &other) 
	{ 
		return Get64() != other.Get64(); 
	}
	
	inline Address & operator +=(unsigned long long off) 
	{ 
		long_addr.ull += off;
		return *this; 
	}
	
private:
	union {
	    unsigned long long ull;
	    unsigned int ui[2];
	} long_addr;
};

}
}
#endif /*ADDRESS_H_*/

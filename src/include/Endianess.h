#ifndef ENDIANESS_H_
#define ENDIANESS_H_

/**
 *  Endian detection support
 */
enum Endianness {
	End_BIG_ENDIAN = 0,
	End_LITTLE_ENDIAN = 1
};

inline Endianness
GetEndianess( void)
{
  uint16 tmp = 1; // for endian testing
  uint8 *ptr = (uint8 *)&tmp;
    
  if( ptr[0] == 1)
    return End_LITTLE_ENDIAN;
  else
    return End_BIG_ENDIAN;
}

inline void Swap2Byte(uint16 what)
{
	uint16 tmp;
	uint8 *ptrSrc = (uint8*)&tmp;
	uint8 *ptrDest = (uint8*)&what;
	
	ptrDest[0] = ptrSrc[1];
	ptrDest[1] = ptrSrc[0];
}

inline void Swap4Byte(uint32 what)
{
	uint32 tmp;
	uint8 *ptrSrc = (uint8*)&tmp;
	uint8 *ptrDest = (uint8*)&what;
	
	ptrDest[0] = ptrSrc[3];
	ptrDest[1] = ptrSrc[2];
	ptrDest[2] = ptrSrc[1];
	ptrDest[3] = ptrSrc[0];
}

inline void Swap8Byte(uint64 what)
{
	uint64 tmp;
	uint8 *ptrSrc = (uint8*)&tmp;
	uint8 *ptrDest = (uint8*)&what;
	
	ptrDest[0] = ptrSrc[7];
	ptrDest[1] = ptrSrc[6];
	ptrDest[2] = ptrSrc[5];
	ptrDest[3] = ptrSrc[4];
	ptrDest[4] = ptrSrc[3];
	ptrDest[5] = ptrSrc[2];
	ptrDest[6] = ptrSrc[1];
	ptrDest[7] = ptrSrc[0];
}

#endif /*ENDIANESS_H_*/

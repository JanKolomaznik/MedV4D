#include "M4DCommon.h"

template<>
int GetNumericTypeID<signed char>()
{ return NTID_SIGNED_CHAR; }

template<>
int GetNumericTypeID<unsigned char>()
{ return NTID_UNSIGNED_CHAR; }

template<>
int GetNumericTypeID<short>()
{ return NTID_SHORT; }

template<>
int GetNumericTypeID<unsigned short>()
{ return NTID_UNSIGNED_SHORT; }

template<>
int GetNumericTypeID<int>()
{ return NTID_INT; }

template<>
int GetNumericTypeID<unsigned int>()
{ return NTID_UNSIGNED_INT; }

template<>
int GetNumericTypeID<long>()
{ return NTID_LONG; }

template<>
int GetNumericTypeID<unsigned long>()
{ return NTID_UNSIGNED_LONG; }

template<>
int GetNumericTypeID<long long>()
{ return NTID_LONG_LONG; }

template<>
int GetNumericTypeID<unsigned long long>()
{ return NTID_UNSIGNED_LONG_LONG; }

template<>
int GetNumericTypeID<float>()
{ return NTID_FLOAT; }

template<>
int GetNumericTypeID<double>()
{ return NTID_DOUBLE; }

template<>
int GetNumericTypeID<bool>()
{ return NTID_BOOL; }
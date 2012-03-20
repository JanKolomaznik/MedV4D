#ifndef _STRING_TOOLS_H
#define _STRING_TOOLS_H

#include <string>
#include <iomanip>
#include <sstream>

/**
 *  @ingroup common
 *  @file StringTools.h
 *
 *  @addtogroup common
 *  @{
 *  @section stringtools String Tools
 */



//TODO test and move to better place
#define MAKESTRING( S ) #S

/**
 * This class enables converting many arguments to a string.
 **/
class ToString
{
public:
  inline operator std::string() const
  {
       return toString();
  }

  inline std::string toString() const
  {
       return m_ostream.str();
  }

  /** Add argument of any type to the stream. */
  template <class ArgType>
  ToString& operator<<(ArgType const& arg)
  {
       m_ostream << arg;
       return *this;
  }

private:
  std::ostringstream m_ostream;
};


/**
Macro for a better usage of the class defined above.
*/
#define TO_STRING(MSG) ( std::string(ToString() << MSG) )

inline std::string
bytesToHumanReadableFormat( size_t aBytes ) 
{
	if ( (aBytes >> 20) > 0 ) {
		return TO_STRING( float( aBytes ) / (1024*1024) << " MB" );
	}
	if ( (aBytes >> 10) > 0 ) {
		return TO_STRING( float( aBytes ) / (1024) << " KB" );
	}
	return TO_STRING( aBytes << " bytes" );
}

/** @} */

#endif /*_STRING_TOOLS_H*/

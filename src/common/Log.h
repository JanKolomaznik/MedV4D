#ifndef __LOG_H_
#define __LOG_H_

#include <ostream>
#include <iomanip>

/**
 *  @ingroup common
 *  @file Log.h
 *
 *  @addtogroup common
 *  @{
 *  @section logging Logging
 *
 *  We have prepared few macros for easy sending informations 
 *  to log stream and few classes designed for formating output.
 *	This hierarchy is still opened and prepared for extending.
 */

extern std::ostream *logStream;

#define LOUT (*logStream)
#define SET_LOUT( N_OSTREAM )	(logStream = &(N_OSTREAM))

class LogFormatter;

std::ostream &
operator<<( std::ostream & stream, const LogFormatter & logFormater );



class LogFormatter
{
public:
	virtual
	~LogFormatter() { }

	virtual void
	Apply( std::ostream &stream ) const = 0;
};

class LogDelimiter: public LogFormatter
{
public:
	LogDelimiter( char delimiter = '-', unsigned length = 80 ) : 
		_delimiter( delimiter ), _length( length ) { }

	void
	Apply( std::ostream &stream ) const;
protected:
	char		_delimiter;
	unsigned	_length;
};

class LogCurrentTime: public LogFormatter
{
public:
	LogCurrentTime() {}
	void
	Apply( std::ostream &stream ) const;
};

#define LOG( ARGs )	\
	LOUT << ARGs << std::endl;

#define LOG_CONT( ARGs )	\
	LOUT << ARGs; LOUT.flush();

/** @} */

#endif /*__LOG_H_*/



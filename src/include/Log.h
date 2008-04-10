#ifndef __LOG_H_
#define __LOG_H_

#include <ostream>
#include <iomanip>

/*extern std::ostream *plog;
#define LOG	(*plog)*/

extern std::ostream *logStream;

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
	(*logStream) << ARGs << std::endl;

#endif /*__LOG_H_*/

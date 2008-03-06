#ifndef __LOG_H_
#define __LOG_H_

#include <ostream>

/*extern std::ostream *plog;
#define LOG	(*plog)*/

extern std::ostream *logStream;

class LogFormater;

std::ostream &
operator<<( std::ostream & stream, const LogFormater & logFormater );



class LogFormater
{
public:
	virtual
	~LogFormater() { }

	virtual void
	Apply( std::ostream &stream ) const = 0;
};

class LogDelimiter: public LogFormater
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



#define LOG( ARGs )	\
	(*logStream) << ARGs;

#endif /*__LOG_H_*/

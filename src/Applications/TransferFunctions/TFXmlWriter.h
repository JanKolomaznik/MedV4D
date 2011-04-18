#ifndef TF_XMLWRITER
#define TF_XMLWRITER

#include <QtCore/QXmlStreamWriter>
#include <QtCore/QString>
#include <QtCore/QFile>

namespace M4D {
namespace GUI {
namespace TF {

class XmlWriterInterface{

public:

	virtual ~XmlWriterInterface();

	virtual bool begin(const std::string& file);

	virtual void writeDTD(const std::string& dtd);

	virtual void writeStartElement(const std::string& element);
	virtual void writeEndElement();

	virtual void writeAttribute(const std::string& attribute, const std::string& value);

	virtual void end();

	bool error(){

		return error_;
	}

	std::string errorMessage(){

		return errorMsg_;
	}

protected:

	bool error_;
	std::string errorMsg_;
	
	XmlWriterInterface():
		error_(false),
		errorMsg_(""){
	}
};

class QtXmlWriter: public XmlWriterInterface{

public:

	QtXmlWriter();
	QtXmlWriter(const std::string& file);
	~QtXmlWriter();

	bool begin(const std::string& file);

	void writeDTD(const std::string& dtd);

	void writeStartElement(const std::string& element);
	void writeEndElement();

	void writeAttribute(const std::string& attribute, const std::string& value);

	void end();

private:

	QXmlStreamWriter qWriter_;
	QFile qFile_;
};

}	//namespace TF
}	//namespace GUI
}	//namespace M4D

#endif	//TF_XMLWRITER
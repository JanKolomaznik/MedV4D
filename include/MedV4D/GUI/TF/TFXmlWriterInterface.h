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

	virtual ~XmlWriterInterface(){}

	virtual bool begin(const std::string& file) = 0;

	virtual void writeDTD(const std::string& dtd) = 0;

	virtual void writeStartElement(const std::string& element) = 0;
	virtual void writeEndElement() = 0;

	virtual void writeAttribute(const std::string& attribute, const std::string& value) = 0;

	virtual void end() = 0;

	std::string fileName(){

		return fileName_;
	}

	bool error(){

		return error_;
	}

	std::string errorMessage(){

		return errorMsg_;
	}

protected:

	bool error_;
	std::string errorMsg_;

	std::string fileName_;
	
	XmlWriterInterface():
		error_(false),
		errorMsg_(""),
		fileName_(""){
	}
};

}	//namespace TF
}	//namespace GUI
}	//namespace M4D

#endif	//TF_XMLWRITER
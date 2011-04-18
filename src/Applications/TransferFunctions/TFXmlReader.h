#ifndef TF_XMLREADER
#define TF_XMLREADER

#include <QtCore/QXmlStreamReader>
#include <QtCore/QString>
#include <QtCore/QFile>

namespace M4D {
namespace GUI {
namespace TF {

class XmlReaderInterface{

public:

	virtual ~XmlReaderInterface(){}

	virtual bool begin(const std::string& file) = 0;

	virtual bool readElement(const std::string& element) = 0;

	virtual std::string readAttribute(const std::string& attribute, const std::string& value) = 0;

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

class QtXmlReader: public XmlReaderInterface{

public:

	QtXmlReader();
	~QtXmlReader();

	bool begin(const std::string& file);
	void end();

	bool readElement(const std::string& element);
	std::string readAttribute(const std::string& attribute);

private:

	QFile file_;
	QXmlStreamReader qReader_;

	bool fileError_;
};

}	//namespace TF
}	//namespace GUI
}	//namespace M4D

#endif //TF_XMLREADER
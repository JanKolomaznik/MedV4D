#ifndef TF_XMLWRITER_QT
#define TF_XMLWRITER_QT

#include "MedV4D/GUI/TF/XmlWriterInterface.h"

namespace M4D {
namespace GUI {
namespace TF {

class QtXmlWriter: public XmlWriterInterface{

public:

	QtXmlWriter();
	~QtXmlWriter();

	bool begin(const std::string& file);
	void end();

	void writeDTD(const std::string& dtd);
	void writeStartElement(const std::string& element);
	void writeEndElement();
	void writeAttribute(const std::string& attribute, const std::string& value);

private:

	QXmlStreamWriter qWriter_;
	QFile qFile_;
};

}	//namespace TF
}	//namespace GUI
}	//namespace M4D

#endif	//TF_XMLWRITER_QT

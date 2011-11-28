#ifndef TF_XMLREADER_QT
#define TF_XMLREADER_QT

#include "MedV4D/GUI/TF/TFXmlReaderInterface.h"

namespace M4D {
namespace GUI {
namespace TF {

class QtXmlReader: public XmlReaderInterface{

public:

	QtXmlReader();
	~QtXmlReader();

	bool begin(const std::string& file);
	void end();

	bool readElement(const std::string& element);
	std::string readAttribute(const std::string& attribute);

private:

	QFile qFile_;
	QXmlStreamReader qReader_;

	bool fileError_;
};

}	//namespace TF
}	//namespace GUI
}	//namespace M4D

#endif //TF_XMLREADER_QT

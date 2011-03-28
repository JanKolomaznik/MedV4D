#ifndef TF_XMLREADER
#define TF_XMLREADER

#include <QtCore/QXmlStreamReader>
#include <QtCore/QString>
#include <QtCore/QFile>

#include <TFCommon.h>

namespace M4D {
namespace GUI {

class TFXmlReader{

public:

	typedef boost::shared_ptr<TFXmlReader> Ptr;

	TFXmlReader(QFile* file);
	~TFXmlReader();

	bool readElement(const std::string& element);
	std::string readAttribute(const std::string& attribute);

	QString fileName();

private:

	QXmlStreamReader qReader_;
	QString fileName_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_XMLREADER
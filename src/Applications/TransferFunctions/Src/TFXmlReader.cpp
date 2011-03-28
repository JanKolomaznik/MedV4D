#include <TFXmlReader.h>

namespace M4D {
namespace GUI {

TFXmlReader::TFXmlReader(QFile *file):
	qReader_(file),
	fileName_(file->fileName()){
}

TFXmlReader::~TFXmlReader(){}

bool TFXmlReader::readElement(const std::string& element){

	while(!qReader_.atEnd())
	{
		qReader_.readNext(); 

		if (qReader_.isStartElement() &&
			(qReader_.name().toString().toStdString() == element)) 
		{
			return true;
		}
	}
	return false;
}

std::string TFXmlReader::readAttribute(const std::string& attribute){

	return qReader_.attributes().value(QString::fromStdString(attribute)).toString().toStdString();
}

QString TFXmlReader::fileName(){

	return fileName_;
}

} // namespace GUI
} // namespace M4D

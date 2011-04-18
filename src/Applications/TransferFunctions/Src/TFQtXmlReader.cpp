#include <TFQtXmlReader.h>

namespace M4D {
namespace GUI {
namespace TF {

QtXmlReader::QtXmlReader(){

	fileError_ = true;
	error_ = true;
	errorMsg_ = "No file assigned.";
}

QtXmlReader::~QtXmlReader(){

	if(qFile_.isOpen()) qFile_.close();
}

bool QtXmlReader::begin(const std::string& file){

	if(file.empty())
	{
		fileError_ = true;
		error_ = true;
		errorMsg_ = "Empty file name.";
		return false;
	}
	
	qFile_.setFileName(QString::fromStdString(file));
	if (!qFile_.open(QFile::ReadOnly | QFile::Text))
	{
		fileError_ = true;
		error_ = true;
		errorMsg_ = "Cannot read file " + file + ":\n"
			+ qFile_.errorString().toStdString() + ".";
		return false;
	}
	fileName_ = file;

	qReader_.setDevice(&qFile_);

	return true;
}

void QtXmlReader::end(){

	if(fileError_) return;
	qFile_.close();
}

bool QtXmlReader::readElement(const std::string& element){

	if(fileError_) return false;

	while(!qReader_.atEnd())
	{
		qReader_.readNext(); 

		if (qReader_.isStartElement() &&
			(qReader_.name().toString().toStdString() == element)) 
		{
			return true;
		}
	}
	if(qReader_.hasError())
	{
		error_ = true;
		errorMsg_ = qReader_.errorString().toStdString();
	}
	return false;
}

std::string QtXmlReader::readAttribute(const std::string& attribute){

	if(fileError_) return false;

	QStringRef value = qReader_.attributes().value(QString::fromStdString(attribute));

	if(value.isEmpty())
	{
		error_ = true;
		errorMsg_ = "Attribute not defined.";
		return "";
	}

	return value.toString().toStdString();
}

}	//namespace TF
}	//namespace GUI
}	//namespace M4D

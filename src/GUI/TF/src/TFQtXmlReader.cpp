#include "GUI/TF/TFQtXmlReader.h"

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
	
	qFile_.setFileName(file.c_str());
	if (!qFile_.open(QFile::ReadOnly | QFile::Text))
	{
		fileError_ = true;
		error_ = true;
		errorMsg_ = "Cannot read file " + file + ":\n"
			+ qFile_.errorString().toLocal8Bit().data() + ".";
		return false;
	}
	fileName_ = file;

	fileError_ = false;
	error_ = false;
	errorMsg_ = "";

	qReader_.setDevice(&qFile_);

	return true;
}

void QtXmlReader::end(){

	if(fileError_) return;
	qFile_.close();
	fileError_ = true;
	error_ = true;
	errorMsg_ = "No file assigned.";
}

bool QtXmlReader::readElement(const std::string& element){

	if(fileError_) return false;

	while(!qReader_.atEnd())
	{
		qReader_.readNext(); 

		if (qReader_.isStartElement() &&
			(qReader_.name().toString().toLocal8Bit().data() == element)) 
		{
			return true;
		}
	}
	if(qReader_.hasError())
	{
		error_ = true;
		errorMsg_ = qReader_.errorString().toLocal8Bit().data();
	}
	return false;
}

std::string QtXmlReader::readAttribute(const std::string& attribute){

	if(fileError_) return false;

	QStringRef value = qReader_.attributes().value(attribute.c_str());

	if(value.isEmpty())
	{
		error_ = true;
		errorMsg_ = "Attribute not defined.";
		return "";
	}

	return value.toString().toLocal8Bit().data();
}

}	//namespace TF
}	//namespace GUI
}	//namespace M4D

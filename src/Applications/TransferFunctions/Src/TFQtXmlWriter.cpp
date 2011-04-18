#include "TFQtXmlWriter.h"

namespace M4D {
namespace GUI {
namespace TF {

QtXmlWriter::QtXmlWriter(){

	error_ = true;
	errorMsg_ = "No file assigned.";
}

QtXmlWriter::~QtXmlWriter(){

	if(qFile_.isOpen()) qFile_.close();
}

bool QtXmlWriter::begin(const std::string& file){

	if(file.empty())
	{
		error_ = true;
		errorMsg_ = "Empty file name.";
		return false;
	}
	
	qFile_.setFileName(QString::fromStdString(file));
	if (!qFile_.open(QFile::WriteOnly | QFile::Text))
	{
		error_ = true;
		errorMsg_ = "Cannot write file " + file + ":\n"
			+ qFile_.errorString().toStdString() + ".";
		return false;
	}
	fileName_ = file;

	qWriter_.setDevice(&qFile_);
	qWriter_.setAutoFormatting(true);
	qWriter_.writeStartDocument();
	return true;
}

void QtXmlWriter::end(){

	if(error_) return;
	qWriter_.writeEndDocument();
	qFile_.close();
}

void QtXmlWriter::writeDTD(const std::string& attribute){

	if(error_) return;
	qWriter_.writeDTD(QString::fromStdString("<!DOCTYPE " + attribute + ">"));
}

void QtXmlWriter::writeStartElement(const std::string& element){

	if(error_) return;
	qWriter_.writeStartElement(QString::fromStdString(element));
}

void QtXmlWriter::writeEndElement(){

	if(error_) return;
	qWriter_.writeEndElement();
}

void QtXmlWriter::writeAttribute(const std::string& attribute, const std::string& value){

	if(error_) return;
	qWriter_.writeAttribute(QString::fromStdString(attribute), QString::fromStdString(value));
}

}	//namespace TF
}	//namespace GUI
}	//namespace M4D

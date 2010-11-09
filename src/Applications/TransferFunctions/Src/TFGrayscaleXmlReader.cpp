#include "TFGrayscaleXmlREADER.h"

namespace M4D {
namespace GUI {

TFGrayscaleXmlREADER::TFGrayscaleXmlREADER() {}

TFGrayscaleXmlREADER::~TFGrayscaleXmlREADER(){}

void TFGrayscaleXmlREADER::read(QIODevice* device, TFGrayscaleFunction* function, bool &error){

	setDevice(device);

	while(!atEnd())
	{
		readNext(); 

		if(isEndElement())
		{
			break;
		}

		if (isStartElement() && (name() == "TransferFunction"))
		{
			readFunction(function, error);
		}
	}
}

void TFGrayscaleXmlREADER::readTestData(TFGrayscaleFunction* function){
	
	TFFunctionMapPtr f = function->getFunction();
	TFSize domain = function->getDomain();
	for(TFSize i = 0; i < domain; ++i)
	{
		(*f)[i] = (i/4)/1000.0;
	}
}

void TFGrayscaleXmlREADER::readFunction(TFGrayscaleFunction* function, bool &error){

	if(convert<std::string, TFType>(attributes().value("type").toString().toStdString()) != TFTYPE_GRAYSCALE)
	{
		error = true;
	}

	TFSize domain = convert<std::string, TFSize>(attributes().value("domain").toString().toStdString());
	function = new TFGrayscaleFunction(domain);

	TFFunctionMapPtr points = function->getFunction();
	TFSize counter = 0;
	while (!atEnd())
	{
		readNext();

		if(isEndElement())
		{
			break;
		}

		if (isStartElement() && (name() == "TFPaintingPoint"))
		{
			if(counter > domain)
			{
				error = true;
				break;
			}

			(*points)[counter] = readPoint(error);
			++counter;

			if(error)
			{
				break;
			}
		}
	}
}

float TFGrayscaleXmlREADER::readPoint(bool &error){

	float loaded = convert<std::string,float>( attributes().value("point").toString().toStdString() );

	while (!atEnd())
	{
		readNext();
		
		if(isEndElement())
		{
			break;
		}
	}
	error = atEnd();
	return loaded;
}

} // namespace GUI
} // namespace M4D

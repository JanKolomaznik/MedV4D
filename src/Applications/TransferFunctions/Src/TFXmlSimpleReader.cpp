#include "TFXmlSimpleReader.h"

namespace M4D {
namespace GUI {

TFXmlSimpleReader::TFXmlSimpleReader() {}

TFXmlSimpleReader::~TFXmlSimpleReader(){}

void TFXmlSimpleReader::read(QIODevice* device, TFSimpleFunction* function, bool &error){

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

void TFXmlSimpleReader::readTestData(TFSimpleFunction* function){
	
	TFFunctionMapPtr f = function->getFunction();
	TFSize domain = function->getDomain();
	for(TFSize i = 0; i < domain; ++i)
	{
		(*f)[i] = (i/4)/1000.0;
	}
}

void TFXmlSimpleReader::readFunction(TFSimpleFunction* function, bool &error){

	if(convert<std::string, TFType>(attributes().value("type").toString().toStdString()) != TFTYPE_SIMPLE)
	{
		error = true;
	}

	TFSize domain = convert<std::string, TFSize>(attributes().value("domain").toString().toStdString());
	function = new TFSimpleFunction(domain);

	TFFunctionMapPtr points = function->getFunction();
	TFSize counter = 0;
	while (!atEnd())
	{
		readNext();

		if(isEndElement())
		{
			break;
		}

		if (isStartElement() && (name() == "TFPointSimple"))
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

float TFXmlSimpleReader::readPoint(bool &error){

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

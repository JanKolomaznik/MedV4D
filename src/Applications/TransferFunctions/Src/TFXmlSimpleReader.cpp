#include "TFXmlSimpleReader.h"


TFXmlSimpleReader::TFXmlSimpleReader() {}

TFXmlSimpleReader::~TFXmlSimpleReader(){}

TFSimpleFunction TFXmlSimpleReader::read(QIODevice* device, bool &error){

	setDevice(device);
	
	TFSimpleFunction loaded;

	while(!atEnd())
	{
		readNext(); 

		if(isEndElement())
		{
			break;
		}

		if (isStartElement() && (name() == "TransferFunction"))
		{
			loaded = readFunction(error);
		}
	}
	return loaded;
}

TFSimpleFunction TFXmlSimpleReader::readFunction(bool &error){

	if(convert<std::string, TFType>(attributes().value("type").toString().toStdString()) != TFTYPE_SIMPLE)
	{
		error = true;
		return TFSimpleFunction();
	}

	TFSimpleFunction loaded( attributes().value("name").toString().toStdString(),
		convert<std::string, int>(attributes().value("functionRange").toString().toStdString()),
		convert<std::string, int>(attributes().value("colorRange").toString().toStdString()) );

	while (!atEnd())
	{
		readNext();

		if(isEndElement())
		{
			break;
		}

		if (isStartElement() && (name() == "TFPoint"))
		{
			TFPoint point = readPoint(error);
			loaded.addPoint(point);

			if(error)
			{
				break;
			}
		}
	}
	return loaded;
}

TFPoint TFXmlSimpleReader::readPoint(bool &error){

	TFPoint loaded(
		convert<std::string,int>( attributes().value("x").toString().toStdString() ),
		convert<std::string,int>( attributes().value("y").toString().toStdString() ) );

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

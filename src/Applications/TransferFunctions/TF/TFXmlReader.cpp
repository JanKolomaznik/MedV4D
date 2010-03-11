#include "TFXmlReader.h"


TFXmlReader::TFXmlReader() {}

TFXmlReader::~TFXmlReader(){}

bool TFXmlReader::read(QIODevice* device, TFScheme** storage){

	setDevice(device);
	
	bool ok = true;

	while(!atEnd())
	{
		readNext(); 

		if(isEndElement())
		{
			break;
		}

		if (isStartElement() || name() == "TFScheme")
		{
			ok = readScheme(storage);
		}
	}

	return ok && !error();
}

bool TFXmlReader::readScheme(TFScheme** scheme){

	*scheme = new TFScheme( attributes().value("name").toString().toStdString() );

	bool ok= true;

	while (!atEnd())
	{
		readNext();  

		if(isEndElement())
		{
			break;
		}

		if (isStartElement() || name() == "TFSchemeFunction")
		{
			TFSchemeFunction* loaded = NULL;
			ok = readFunction(&loaded);
			(*scheme)->addFunction(loaded);

			if(!ok)
			{
				break;
			}
		}
	}

	return ok && !error();
}

bool TFXmlReader::readFunction(TFSchemeFunction** function){

	*function = new TFSchemeFunction(
		attributes().value("name").toString().toStdString(),
		attributes().value("colourR").toString().toInt(),
		attributes().value("colourG").toString().toInt(),
		attributes().value("colourB").toString().toInt() );

	bool ok = true;

	while (!atEnd())
	{
		readNext();

		if(isEndElement())
		{
			break;
		}

		if (isStartElement() || name() == "TFSchemePoint")
		{
			TFSchemePoint* loaded = NULL;
			ok = readPoint(&loaded);
			(*function)->addPoint(loaded);

			if(!ok)
			{
				break;
			}
		}
	}

	return ok && !error();
}

bool TFXmlReader::readPoint(TFSchemePoint** point){

	*point = new TFSchemePoint(
		convert<string,int>( attributes().value("x").toString().toStdString() ),
		convert<string,int>( attributes().value("y").toString().toStdString() ) );

	while (!atEnd())
	{
		readNext();
		
		if(isEndElement())
		{
			break;
		}
	}

	return !atEnd() && !error();
}

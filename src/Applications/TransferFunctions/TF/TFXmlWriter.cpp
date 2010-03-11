#include "TFXmlWriter.h"

 TFXmlWriter::TFXmlWriter(){
     setAutoFormatting(true);
 }

 bool TFXmlWriter::write(QIODevice *device, TFScheme** data){

     setDevice(device);

     writeStartDocument();
     writeDTD("<!DOCTYPE TFScheme>");

	 writeScheme(data);
	
     writeEndDocument();
     
	 return true;
 }

 void TFXmlWriter::writeScheme(TFScheme** scheme){

     writeStartElement("TFScheme");
	 writeAttribute("name", QString::fromStdString((*scheme)->name));

	vector<TFName> points = (*scheme)->getFunctionNames();
	vector<TFName>::iterator first = points.begin();
	vector<TFName>::iterator end = points.end();
	vector<TFName>::iterator it = first;

     for(it; it != end; ++it)
	 {
		 TFSchemeFunction* f = (*scheme)->getFunction(*it);
		 writeFunction(&f);
		 delete f;
	 }
         
	 writeEndElement();
 }

 void TFXmlWriter::writeFunction(TFSchemeFunction** function){

     writeStartElement("TFSchemeFunction");
	 writeAttribute("name", QString::fromStdString((*function)->name));
     writeAttribute("colourR", QString::fromStdString( convert<int, string>((*function)->colourRGB[0]) ));
     writeAttribute("colourG", QString::fromStdString( convert<int, string>((*function)->colourRGB[1]) ));
     writeAttribute("colourB", QString::fromStdString( convert<int, string>((*function)->colourRGB[2]) ));

	vector<TFSchemePoint*> points = (*function)->getAllPoints();
	vector<TFSchemePoint*>::iterator first = points.begin();
	vector<TFSchemePoint*>::iterator end = points.end();
	vector<TFSchemePoint*>::iterator it = first;

     for(it; it != end; ++it)
	 {
         writePoint(&(*it));
		 delete *it;
	 }

     writeEndElement();
 }

 void TFXmlWriter::writePoint(TFSchemePoint** point){

     writeStartElement("TFSchemePoint");
     writeAttribute("x", QString::fromStdString( convert<int, string>((*point)->x)) );
     writeAttribute("y", QString::fromStdString( convert<int, string>((*point)->y)) );

     writeEndElement();	//jednorazovy element?
 }
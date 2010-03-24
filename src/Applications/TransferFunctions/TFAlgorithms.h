#ifndef TF_ALGORITHMS
#define TF_ALGORITHMS

#include <QtCore/QString>
#include <QtCore/QFile>
#include <QtCore/QXmlStreamReader>

#include <QtGui/QWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>

#include <fstream>
#include <cassert>

#include <TFSimpleHolder.h>


class TFHolderFactory{

public:
	
	static TFActions createMenuTFActions(QWidget *owner, QMenu *menu){

		TFActions actions;

		actions.push_back(new TFAction(owner, menu, TFTYPE_SIMPLE));

		return actions;
	}

	static TFAbstractHolder* create(TFType &holderType){

		switch(holderType)
		{
			case TFTYPE_SIMPLE:
			{
				return new TFSimpleHolder();
			}
			case TFTYPE_UNKNOWN:
			default:
			{
				assert("unknown holder");
				break;
			}
		}
		return NULL;
	}

	static TFAbstractHolder* load(QWidget* parent){
		
		QString fileName = QFileDialog::getOpenFileName(parent,
			QObject::tr("Open Transfer Function"),
			QDir::currentPath(),
			QObject::tr("TF Files (*.tf *.xml)"));

		if (fileName.isEmpty()) return NULL;

		QFile qFile(fileName);

		if (!qFile.open(QFile::ReadOnly | QFile::Text)) {
			QMessageBox::warning(parent, QObject::tr("Transfer Functions"),
							  QObject::tr("Cannot read file %1:\n%2.")
							  .arg(fileName)
							  .arg(qFile.errorString()));
			return NULL;
		}

		TFTypeSwitcher switcher;
		TFType tfType = switcher.read(&qFile);
		qFile.close();

		qFile.open(QFile::ReadOnly | QFile::Text);
		TFAbstractHolder* loaded = NULL;
		switch(tfType)
		{
			case TFTYPE_SIMPLE:
			{
				loaded = new TFSimpleHolder();
				if(!loaded->_load(qFile))
				{ 
					QMessageBox::warning(parent,
						QObject::tr("TFXmlSimpleReader"),
						QObject::tr("Parse error in file %1").arg(fileName));
				}
				break;
			}
			case TFTYPE_UNKNOWN:
			default:
			{
				assert("unknown holder");
				break;
			}
		}
		qFile.close();
		return loaded;
	}

protected:

	class TFTypeSwitcher: public QXmlStreamReader{

	public:
		TFTypeSwitcher(){}
		~TFTypeSwitcher(){}

		TFType read(QIODevice* device){

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
					return convert<std::string, TFType>(attributes().value("type").toString().toStdString());
				}
			}
			return TFTYPE_UNKNOWN;
		}
	};

private:
	TFHolderFactory(){}
	~TFHolderFactory(){}
};

template<typename Iterator>
std::vector<typename std::iterator_traits<Iterator>::value_type> adjustByTransferFunction(
	typename std::iterator_traits<Iterator>::value_type min,
	typename std::iterator_traits<Iterator>::value_type max,
	TFAbstractFunction &transferFunction){

	std::vector<typename std::iterator_traits<Iterator>::value_type> pointMap;
	switch(transferFunction.getType()){
		case TFTYPE_SIMPLE:
		{
			pointMap = adjustBySimpleFunction<Iterator>(min, max, transferFunction);
			break;
		}
		case TFTYPE_UNKNOWN:
		default:
		{
			assert("Unknown Transfer Function");
			break;
		}
	}
	return pointMap;
}

template<typename Iterator>
std::vector<typename std::iterator_traits<Iterator>::value_type> adjustBySimpleFunction(
	typename std::iterator_traits<Iterator>::value_type min,
	typename std::iterator_traits<Iterator>::value_type max,
	TFAbstractFunction &transferFunction){

	TFSimpleFunction *tf = dynamic_cast<TFSimpleFunction*>(&transferFunction);
	TFPointMap points;	
	std::vector<typename std::iterator_traits<Iterator>::value_type> computed;

	if ( !tf)
	{
		return computed;
	}

	points = tf->getPointMap();

	if(points.empty())
	{
		return computed;
	}

	double range = max - min;
	double interval = range/tf->getFunctionRange();
	typename std::iterator_traits<Iterator>::value_type pixelValue = (points[0]/(double)tf->getColorRange())*range;
	typename std::iterator_traits<Iterator>::value_type nextPixelValue = pixelValue;
	long intervalCorrection = 0;	//problem - cannot be double because % is needed and will not work for types with greater range

	for (int i = 1; i < tf->getFunctionRange(); ++i )
	{
		pixelValue = nextPixelValue;
		nextPixelValue = (points[i]/(double)tf->getColorRange())*range;

		double intervalBottom = (interval + intervalCorrection);
		intervalCorrection = (long)intervalBottom % 1;
		intervalBottom -= intervalCorrection;
		double step = (nextPixelValue - pixelValue);
		if(intervalBottom != 1) step = step/(intervalBottom - 1);

		for(double i = 0; i < intervalBottom; ++i)
		{
			computed.push_back(pixelValue + i*step);
		}
	}

	pixelValue = nextPixelValue;
	nextPixelValue = (points[tf->getFunctionRange()]/(double)tf->getColorRange())*range;
	intervalCorrection = range - computed.size();
	double step = (nextPixelValue - pixelValue);
	if(intervalCorrection != 0) step = step/intervalCorrection;

	for(double i = 0; i < intervalCorrection; ++i)
	{
		computed.push_back(pixelValue + i*step);
	}
	return computed;
}

#endif //TF_ALGORITHMS
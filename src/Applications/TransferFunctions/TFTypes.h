#ifndef TF_TYPES
#define TF_TYPES

#include <sstream>

#include <QtGui/QAction>
#include <QtGui/QMenu>


typedef std::string TFName;
typedef unsigned long TFSize;

struct TFPoint{

	int x;
	int y;

	TFPoint(): x(0), y(0){}
	TFPoint(const TFPoint &point): x(point.x), y(point.y){}
	TFPoint(int x, int y): x(x), y(y){}

	bool operator==(const TFPoint& point){
		return (x == point.x) && (y == point.y);
	}
};

typedef std::vector<TFPoint> TFPoints;
typedef TFPoints::iterator TFPointsIterator;

typedef std::vector<int> TFPointMap;
typedef TFPointMap::iterator TFPointMapIterator;


template<typename From, typename To>
static To convert(const From &s){

	std::stringstream ss;
    To d;
    ss << s;
    if(ss >> d)
	{
        return d;
	}
    return NULL;
}

enum TFType{
	TFTYPE_UNKNOWN,
	TFTYPE_SIMPLE
};

template<>
static std::string convert<TFType, std::string>(const TFType &tfType){

	switch(tfType){
		case TFTYPE_SIMPLE:
		{
			return "Simple";
		}
	}
	return "Unknown";
}

template<>
static TFType convert<std::string, TFType>(const std::string &tfType){

	if(tfType == "Simple"){
		return TFTYPE_SIMPLE;
	}
	return TFTYPE_UNKNOWN;
}

class TFAction: public QObject{

	Q_OBJECT

public:
	TFAction(QMenu* menu, TFType tfType){
		type_ = tfType;

		QString name = QString::fromStdString(convert<TFType, std::string>(type_));

		action_ = new QAction(menu);
		action_->setObjectName(name);
		action_->setText(name);
		menu->addAction(action_);

		QObject::connect( action_, SIGNAL(triggered()), this, SLOT(triggered()));
	}

	~TFAction(){
		delete action_;
	}

signals:
	void TFActionClicked(TFType &tfType);

public slots:
	void triggered(){
		emit TFActionClicked(type_);
	}

private:	
	QAction* action_;
	TFType type_;
};

typedef std::vector<TFAction*> TFActions;
typedef std::vector<unsigned> TFHistogram;

#endif //TF_TYPES
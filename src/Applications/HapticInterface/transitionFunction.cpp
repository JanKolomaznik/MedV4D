#include "transitionFunction.h"
#include <algorithm>
#include <iostream>

transitionFunction::transitionFunction(unsigned short minPoint, unsigned short maxPoint, double minValue, double maxValue)
{
	pointValue minpv(minPoint, minValue);
	pointValue maxpv(maxPoint, maxValue);
	data.push_back(minpv);
	data.push_back(maxpv);
	solidFrom = -1;
}

void transitionFunction::Reset(unsigned short minPoint, unsigned short maxPoint, double minValue, double maxValue)
{
	boost::mutex::scoped_lock l(accesMutex);
	data.clear();
	pointValue minpv(minPoint, minValue);
	pointValue maxpv(maxPoint, maxValue);
	data.push_back(minpv);
	data.push_back(maxpv);
	solidFrom = -1;
}

bool transitionFunction::pointValue::operator !=(const transitionFunction::pointValue &rhs)
{
	return this->point != rhs.point;
}

bool transitionFunction::pointValue::operator ==(const transitionFunction::pointValue &rhs)
{
	return this->point == rhs.point;
}

bool transitionFunction::pointValue::operator <(const transitionFunction::pointValue &rhs)
{
	return this->point < rhs.point;
}

bool transitionFunction::pointValue::operator >(const transitionFunction::pointValue &rhs)
{
	return this->point > rhs.point;
}

bool transitionFunction::pointValue::operator <=(const transitionFunction::pointValue &rhs)
{
	return this->point <= rhs.point;
}

bool transitionFunction::pointValue::operator >=(const transitionFunction::pointValue &rhs)
{
	return this->point >= rhs.point;
}

unsigned short transitionFunction::GetMaxPoint()
{
	boost::mutex::scoped_lock l(accesMutex);
	return data[data.size() - 1].point;
}

void transitionFunction::SetMaxPoint(unsigned short point)
{
	boost::mutex::scoped_lock l(accesMutex);
	data[data.size() - 1].point = point;
	sort(data.begin(), data.end());
}

unsigned short transitionFunction::GetMinPoint()
{
	boost::mutex::scoped_lock l(accesMutex);
	return data[0].point;
}

void transitionFunction::SetMinPoint(unsigned short point)
{
	boost::mutex::scoped_lock l(accesMutex);
	data[0].point = point;
	sort(data.begin(), data.end());
}

double transitionFunction::GetValueOfMaxPoint()
{
	boost::mutex::scoped_lock l(accesMutex);
	return data[data.size() - 1].val;
}

void transitionFunction::SetValueOfMaxPoint(double val)
{
	boost::mutex::scoped_lock l(accesMutex);
	data[data.size() - 1].val = val;
}

double transitionFunction::GetValueOfMinPoint()
{
	boost::mutex::scoped_lock l(accesMutex);
	return data[0].val;
}

void transitionFunction::SetValueOfMinPoint(double val)
{
	boost::mutex::scoped_lock l(accesMutex);
	data[0].val = val;
}

double transitionFunction::GetValueOnPoint(unsigned short point)
{
	boost::mutex::scoped_lock l(accesMutex);
	//if ((solidFrom != -1) && (point > solidFrom))
	//{
	//	return 1.0;
	//}
	size_t i;
	for (i = 0; i < data.size(); ++i)
	{
		if (data[i].point >= point)
			break;
	}
	if (data[i].point == point)
	{
		return data[i].val;
	}
	unsigned short countBetween = data[i].point - data[i - 1].point;
	double valueDifference = data[i].val - data[i - 1].val;
	double valuePerPoint = valueDifference / (double)countBetween;
	return valuePerPoint * (double)(point - data[i - 1].point) + data[i-1].val;
}

void transitionFunction::SetValueOnPoint(unsigned short point, double val)
{
	boost::mutex::scoped_lock l(accesMutex);
	if (point > data[data.size()- 1].point)
	{
		return;
	}
	size_t i;
	for (i = 0; i < data.size(); ++i)
	{
		if (data[i].point >= point)
			break;
	}
	if (i != data.size())
	{
		if (data[i].point == point)
		{
			data[i].val = val;
		}
		else
		{
			pointValue pv(point, val);
			data.push_back(pv);
			sort(data.begin(), data.end());
		}
	}
	else
	{
		pointValue pv(point, val);
		data.push_back(pv);
		sort(data.begin(), data.end());
	}
}

std::size_t transitionFunction::size()
{
	boost::mutex::scoped_lock l(accesMutex);
	return data.size();
}

unsigned short transitionFunction::GetPointOnPosition( std::size_t pos )
{	
	boost::mutex::scoped_lock l(accesMutex);
	return data[pos].point;
}

void transitionFunction::DeletePointOnPosition(std::size_t position)
{
	boost::mutex::scoped_lock l(accesMutex);
	if (position < data.size())
	{
		for (size_t i = position; i < data.size() - 1; ++i)
		{
			data[i] = data[i + 1];
		}
		data.pop_back();
	}
}

void transitionFunction::SetPointOnPosition( std::size_t pos, unsigned short point )
{
	boost::mutex::scoped_lock l(accesMutex);
	data[pos].point = point;
}

void transitionFunction::SaveToFile( std::string fileName )
{
	boost::mutex::scoped_lock l(accesMutex);
	std::ofstream oFile;
	oFile.open(fileName.c_str());
	oFile << data.size() << std::endl;
	for (int i = 0; i < data.size(); ++i)
	{
		oFile << data[i].point << " " << data[i].val << std::endl;
	}
	oFile << solidFrom << std::endl;
}

void transitionFunction::LoadFromFile( std::string fileName )
{
	boost::mutex::scoped_lock l(accesMutex);
	std::ifstream iFile;
	iFile.open(fileName.c_str());
	size_t dataSize;
	iFile >> dataSize;
	unsigned short point;
	double val;
	data.clear();
	for (size_t i = 0; i < dataSize; ++i)
	{
		iFile >> point;
		iFile >> val;
		data.push_back(pointValue(point, val));
	}
	iFile >> solidFrom;
}

int transitionFunction::GetSolidFrom()
{
	return solidFrom;
}

void transitionFunction::SetSolidFrom( int a_solidFrom )
{
	solidFrom = a_solidFrom;
}
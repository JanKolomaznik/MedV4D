#include "transitionFunction.h"
#include <algorithm>

transitionFunction::transitionFunction(unsigned short minPoint, unsigned short maxPoint, double minValue, double maxValue)
{
	pointValue minpv(minPoint, minValue);
	pointValue maxpv(maxPoint, maxValue);
	data.push_back(minpv);
	data.push_back(maxpv);
}

void transitionFunction::Reset(unsigned short minPoint, unsigned short maxPoint, double minValue, double maxValue)
{
	boost::mutex::scoped_lock l(accesMutex);
	data.clear();
	pointValue minpv(minPoint, minValue);
	pointValue maxpv(maxPoint, maxValue);
	data.push_back(minpv);
	data.push_back(maxpv);
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
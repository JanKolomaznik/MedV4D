#include <QtGui>

#include "transitionFunctionRenderArea.h"
#include <sstream>

transitionFunctionRenderAreaWidget::transitionFunctionRenderAreaWidget(transitionFunction* functionData, QWidget *parent)
    : QWidget(parent)
{
	this->functionData = functionData;
	setMouseTracking(true);
    setBackgroundRole(QPalette::Base);
    setAutoFillBackground(true);
	Qt::PenStyle style = Qt::PenStyle(Qt::PenStyle::SolidLine);
	Qt::PenCapStyle cap = Qt::PenCapStyle(Qt::PenCapStyle::FlatCap);
	Qt::PenJoinStyle join = Qt::PenJoinStyle(Qt::PenJoinStyle::MiterJoin);
	pen = QPen(Qt::black, 1, style, cap, join);
	pointStyleMoveable = false;
	selectedPoint = -1;
	lengthToPickPoint = 8;
	mouseX = 0;
	mouseY = 0;
	deny = false;
}

QSize transitionFunctionRenderAreaWidget::minimumSizeHint() const
{
    return QSize(100, 100);
}

QSize transitionFunctionRenderAreaWidget::sizeHint() const
{
    return QSize(800, 200);
}

void transitionFunctionRenderAreaWidget::paintEvent(QPaintEvent * /* event */)
{
    double xstep = (double)(width() - 2) / (double)(functionData->GetMaxPoint() - functionData->GetMinPoint());
	pen = QPen(Qt::black, 1);

	QPainterPath path;
    path.moveTo((int)(xstep * (double)functionData->GetMinPoint()) + 1,height() - ((int)((double)(height()-2) * functionData->GetValueOfMinPoint()) + 1));
    for (size_t i = 1; i < functionData->size(); ++i)
	{
		path.lineTo(QPoint((int)(xstep * (double)functionData->GetPointOnPosition(i)) + 1, height()-((int)((double)(height()-2) * functionData->GetValueOnPoint(functionData->GetPointOnPosition(i))) + 1)));
	}

    QPainter painter(this);
    painter.setPen(pen);
	painter.setBrush(Qt::NoBrush);
    painter.save();
	painter.drawPath(path);

	painter.restore();

	if (functionData->GetSolidFrom() != -1)
	{
		painter.setPen(QPen(Qt::magenta, 1));
		painter.setBrush(Qt::NoBrush);
		painter.save();
		double xpos = ((double)functionData->GetSolidFrom()) * width() / ((double)functionData->GetMaxPoint());
		painter.drawLine(xpos, 1, xpos, height() - 1);
		painter.restore();
	}

	if (functionData->GetSolidTo() != -1)
	{
		painter.setPen(QPen(Qt::cyan, 1));
		painter.setBrush(Qt::NoBrush);
		painter.save();
		double xpos = ((double)functionData->GetSolidTo()) * width() / ((double)functionData->GetMaxPoint());
		painter.drawLine(xpos, 1, xpos, height() - 1);
		painter.restore();
	}

    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
	painter.save();
    painter.drawRect(QRect(0, 0, width() - 1, height() - 1));

	painter.restore();

	painter.setPen(pen);
	painter.setBrush(Qt::NoBrush);
	painter.save();
	painter.drawText(1, height() - 3, tr("0"));

	painter.restore();

	painter.setPen(pen);
		painter.setBrush(Qt::NoBrush);
	painter.save();
	painter.drawText(1, 11, tr("1.0"));

	painter.restore();

	painter.setPen(pen);
		painter.setBrush(Qt::NoBrush);
	painter.save();
	std::stringstream s;
	s << functionData->GetMaxPoint();
	painter.drawText(width() - 26, height() - 3, tr(s.str().c_str()));

	if (pointStyleMoveable)
	{
		pen = QPen(Qt::green, 2);
		for (size_t i = 0; i < functionData->size(); ++i)
		{
			painter.restore();
			painter.setPen(pen);
			painter.setBrush(Qt::NoBrush);
			painter.save();

			QRect elipse = QRect((int)(xstep * (double)functionData->GetPointOnPosition(i)) + 1 - lengthToPickPoint,
				height()-((int)((double)(height()-2) * functionData->GetValueOnPoint(functionData->GetPointOnPosition(i))) + 1) - lengthToPickPoint,
				lengthToPickPoint * 2, lengthToPickPoint * 2);
			painter.drawEllipse(elipse);
		}
	}
	if (selectedPoint != -1)
	{
		pen = QPen(Qt::blue, 2);
		painter.restore();
		painter.setPen(pen);
		painter.setBrush(Qt::NoBrush);
		painter.save();

		QRect elipse = QRect(mouseX - lengthToPickPoint, mouseY - lengthToPickPoint, lengthToPickPoint * 2, lengthToPickPoint * 2);
		painter.drawEllipse(elipse);

		QPainterPath hlpLines;
		bool possible = true;
		
		if (selectedPoint != 0)
		{
			hlpLines.moveTo(QPoint((int)(xstep * (double)functionData->GetPointOnPosition(selectedPoint - 1)) + 1, height()-((int)((double)(height()-2) * functionData->GetValueOnPoint(functionData->GetPointOnPosition(selectedPoint - 1))) + 1)));
			hlpLines.lineTo(QPoint(mouseX, mouseY));
			if (mouseX <= (int)(xstep * (double)functionData->GetPointOnPosition(selectedPoint - 1)) + 1)
			{
				possible = false;
			}
		}
		else
		{	
			hlpLines.moveTo(QPoint(mouseX, mouseY));
			if (mouseX <= 1)
			{
				possible = false;
			}
		}
		if (selectedPoint < (functionData->size() - 1))
		{
			hlpLines.lineTo(QPoint((int)(xstep * (double)functionData->GetPointOnPosition(selectedPoint + 1)) + 1, height()-((int)((double)(height()-2) * functionData->GetValueOnPoint(functionData->GetPointOnPosition(selectedPoint + 1))) + 1)));
			if (mouseX >= (int)(xstep * (double)functionData->GetPointOnPosition(selectedPoint + 1)) + 1)
			{
				possible = false;
			}
		}
		else
		{
			if (mouseX >= width() - 1)
			{
				possible = false;
			}
		}

		if ((mouseY < 1) || (mouseY > height()))
		{
			possible = false;
		}

		if (possible)
		{
			pen = QPen(Qt::blue, 1);
		}
		else
		{
			pen = QPen(Qt::red, 1);
		}
		painter.restore();
		painter.setPen(pen);
		painter.setBrush(Qt::NoBrush);
		painter.save();
		painter.drawPath(hlpLines);
	}
}

void transitionFunctionRenderAreaWidget::mouseReleaseEvent(QMouseEvent *event)
{
	if ((event->pos().x() < 0) || (event->pos().x() > width()) || (event->pos().y() < 0) || (event->pos().y() > height()))
	{
		return;
	}
	if (event->button() == Qt::LeftButton) 
	{
		if (!pointStyleMoveable)
		{
			double x = ((double)event->pos().x() - 1.0)/ ((double)width() - 2);
			if (x < 0.0)
			{
				x = 0.0;
			}
			double y = ((double)height() - (double)event->pos().y() + 1.0) / ((double)height() - 2.0);
			if (y < 0.0)
			{
				y = 0.0;
			}
			emit addPointSignal(x, y);
		}
		else
		{
			if (selectedPoint == -1)
			{
				if (!deny)
				{
					double x = ((double)event->pos().x() - 1.0)/ ((double)width() - 2);
					if (x < 0.0)
					{
						x = 0.0;
					}
					double y = ((double)height() - (double)event->pos().y() + 1.0) / ((double)height() - 2.0);
					if (y < 0.0)
					{
						y = 0.0;
					}
					emit addPointSignal(x, y);
				}
				deny = false;
			}
			else
			{
				double xstep = (double)(width() - 2) / (double)(functionData->GetMaxPoint() - functionData->GetMinPoint());
				unsigned short x = (int)((event->pos().x() - 1) / xstep);
				double y = ((double)(height() - event->pos().y() + 1) / ((double)(height()-2)));
				if ((x > functionData->GetPointOnPosition(selectedPoint - 1)) && (x < functionData->GetPointOnPosition(selectedPoint + 1)))
				{
					functionData->SetPointOnPosition(selectedPoint, x);
					functionData->SetValueOnPoint(x, y);
				}
				selectedPoint = -1;
			}
		}
	}
	if (event->button() == Qt::RightButton)
	{
		if (selectedPoint != -1)
		{
			selectedPoint = -1;
			deny =  true;
		}
		else
		{
			if (pointStyleMoveable)
			{
				double xstep = (double)(width() - 2) / (double)(functionData->GetMaxPoint() - functionData->GetMinPoint());
				int minx = (int)((event->pos().x() - 1 - lengthToPickPoint) / xstep);
				int maxx = (int)((event->pos().x() - 1 + lengthToPickPoint) / xstep);
				double miny = ((double)(height() - event->pos().y() + 1 - lengthToPickPoint) / ((double)(height()-2)));
				double maxy = ((double)(height() - event->pos().y() + 1 + lengthToPickPoint) / ((double)(height()-2)));
				int i;
				bool chosed = false;
				for (i = 0; i < functionData->size(); ++i)
				{
					if ((minx < functionData->GetPointOnPosition(i)) && (maxx > functionData->GetPointOnPosition(i)))
					{
						chosed = true;
						break;
					}
				}
				if (chosed)
				{
					unsigned short point = functionData->GetPointOnPosition(i);
					if ((miny < functionData->GetValueOnPoint(point)) && (maxy > functionData->GetValueOnPoint(point)))
					{
						if ((i != 0) && (i != functionData->size() - 1))
						{
							functionData->DeletePointOnPosition(i);
						}
					}
				}
			}
		}
	}
	update();
}

void transitionFunctionRenderAreaWidget::mouseMoveEvent( QMouseEvent * event )
{
	mouseX = event->pos().x();
	mouseY = event->pos().y();
	double x = (double)mouseX / (double)width();
	double y = ((double)height() - (double)mouseY) / (double)height();
	emit mouseCoordinatesChangedSignal(x, y);
	update();
}

void transitionFunctionRenderAreaWidget::leaveEvent( QEvent *event )
{
	selectedPoint = -1;
	deny = true;
}

void transitionFunctionRenderAreaWidget::stateChangedSlot( int state )
{
	if (state == 0)
	{
		pointStyleMoveable = false;
		selectedPoint = -1;
	}
	else
	{
		pointStyleMoveable = true;
	}
	update();
}

void transitionFunctionRenderAreaWidget::mousePressEvent( QMouseEvent * event )
{
	deny = false;
	if (pointStyleMoveable)
	{
		if (event->button() == Qt::LeftButton)
		{
			double xstep = (double)(width() - 2) / (double)(functionData->GetMaxPoint() - functionData->GetMinPoint());
			int minx = (int)((event->pos().x() - 1 - lengthToPickPoint) / xstep);
			int maxx = (int)((event->pos().x() - 1 + lengthToPickPoint) / xstep);
			double miny = ((double)(height() - event->pos().y() + 1 - lengthToPickPoint) / ((double)(height()-2)));
			double maxy = ((double)(height() - event->pos().y() + 1 + lengthToPickPoint) / ((double)(height()-2)));
			int i;
			bool chosed = false;
			for (i = 0; i < functionData->size(); ++i)
			{
				if ((minx < functionData->GetPointOnPosition(i)) && (maxx > functionData->GetPointOnPosition(i)))
				{
					chosed = true;
					break;
				} 
			}
			if (chosed)
			{
				unsigned short point = functionData->GetPointOnPosition(i);
				if ((miny < functionData->GetValueOnPoint(point)) && (maxy > functionData->GetValueOnPoint(point)))
				{
					if ((i != 0) && (i != functionData->size() - 1))
					{
						selectedPoint = i;
					}
				}
			}
		}
	}
}
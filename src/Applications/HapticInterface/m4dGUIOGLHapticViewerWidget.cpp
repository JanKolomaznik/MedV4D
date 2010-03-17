/**
*  @ingroup gui
*  @file m4dGUIOpenGLViewerWidget.cpp
*  @brief some brief
*/

#include "m4dGUIOGLHapticViewerWidget.h"
#include <sstream>
#include <QtGui>
#include <string>
#include <queue>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

namespace M4D
{
	namespace Viewer
	{
		m4dGUIOGLHapticViewerWidget::m4dGUIOGLHapticViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent) : QGLWidget(parent)
		{
			std::vector< int > row;
			std::vector< std::vector < int > > layer;
			row.push_back(0);
			layer.push_back(row);
			myData.push_back(layer);
			graphicData.push_back(layer);
			_index = index;
			inPort = new Imaging::InputPortTyped<Imaging::AImage>();
			_inputPorts.AppendPort( inPort );
			//cursor = new cursorInterface(inPort);
			cursor = new hapticCursor(inPort);
			setInputPort( conn );
			preprocessData();
		}
		m4dGUIOGLHapticViewerWidget::m4dGUIOGLHapticViewerWidget(unsigned int index, QWidget *parent) : QGLWidget(parent)
		{
			std::vector< int > row;
			std::vector< std::vector < int > > layer;
			row.push_back(0);
			layer.push_back(row);
			myData.push_back(layer);
			graphicData.push_back(layer);
			_index = index;
			inPort = new Imaging::InputPortTyped<Imaging::AImage>();
			_inputPorts.AppendPort( inPort );
			//cursor = new cursorInterface(inPort);
			cursor = new hapticCursor(inPort);
			setInputPort( );
			preprocessData();
		}
		m4dGUIOGLHapticViewerWidget::~m4dGUIOGLHapticViewerWidget()
		{
		}
		void m4dGUIOGLHapticViewerWidget::setInputPort()
		{
			inPort->UnPlug();
			updateGL();
		}
		void m4dGUIOGLHapticViewerWidget::setInputPort( Imaging::ConnectionInterface* conn )
		{
			if ( !conn )
			{
				setInputPort();
				return;
			}
			conn->ConnectConsumer( *inPort );
			updateGL();
		}
		void m4dGUIOGLHapticViewerWidget::setUnSelected()
		{

		}
		void m4dGUIOGLHapticViewerWidget::setSelected()
		{

		}
		m4dGUIOGLHapticViewerWidget::AvailableSlots m4dGUIOGLHapticViewerWidget::getAvailableSlots()
		{
			return _availableSlots;
		}
		QWidget* m4dGUIOGLHapticViewerWidget::operator ()()
		{
			return (QGLWidget*)this;
		}
		void m4dGUIOGLHapticViewerWidget::ReceiveMessage( Imaging::PipelineMessage::Ptr msg, Imaging::PipelineMessage::MessageSendStyle sendStyle, Imaging::FlowDirection direction )
		{
			emit signalMessageHandler( msg->msgID );
		}
		void m4dGUIOGLHapticViewerWidget::mousePressEvent(QMouseEvent *event)
		{
			if (_eventHandler)
			{
				_eventHandler->mousePressEvent(event);
				return;
			}
			lastMousePosition = QPoint(event->x(), event->y());
			rightButton = event->buttons() == Qt::RightButton;
			leftButton = event->buttons() == Qt::LeftButton;
			updateGL();

		}
		void m4dGUIOGLHapticViewerWidget::mouseReleaseEvent(QMouseEvent *event)
		{
			if (_eventHandler)
			{
				_eventHandler->mouseReleaseEvent(event);
				return;
			}
			rightButton = rightButton && !(event->buttons() == Qt::RightButton);
			leftButton = leftButton && !(event->buttons() == Qt::LeftButton);
			updateGL();
		}
		void m4dGUIOGLHapticViewerWidget::mouseMoveEvent(QMouseEvent *event)
		{
			if (_eventHandler)
			{
				_eventHandler->mouseMoveEvent(event);
				return;
			}
			if (leftButton)
			{
				float xMove = ((float)(event->x() - lastMousePosition.x())) / ((float)(height()));
				float yMove = ((float)(event->y() - lastMousePosition.y())) / ((float)(width()));

				rotateY += xMove * 5.0;
				rotateY = rotateY > 360.0 ? rotateY - 360.0 : rotateY;
				rotateY = rotateY < -360.0 ? rotateY + 360.0 : rotateY;

				rotateX += yMove * 5.0; // TODO
				rotateX = rotateX > 360.0 ? rotateX - 360.0 : rotateX;
				rotateX = rotateX < 360.0 ? rotateX + 360.0 : rotateX;
			}
			updateGL();
		}
		void m4dGUIOGLHapticViewerWidget::wheelEvent(QWheelEvent *event)
		{
			if (_eventHandler)
			{
				_eventHandler->wheelEvent(event);
				return;
			}
			int numDegrees = event->delta() / 8;
			int numSteps = numDegrees / 15;

			zoom += numSteps;

			updateGL();
		}
		void m4dGUIOGLHapticViewerWidget::mouseDoubleClickEvent ( QMouseEvent * event )
		{
			if (_eventHandler)
			{
				_eventHandler->mouseDoubleClickEvent(event);
				return;
			}
		}
		void m4dGUIOGLHapticViewerWidget::keyPressEvent ( QKeyEvent * event )
		{
			if (_eventHandler)
			{
				_eventHandler->keyPressEvent(event);
				return;
			}
		}
		void m4dGUIOGLHapticViewerWidget::keyReleaseEvent ( QKeyEvent * event )
		{
			if (_eventHandler)
			{
				_eventHandler->keyReleaseEvent(event);
				return;
			}
		}
		void m4dGUIOGLHapticViewerWidget::DrawBlock( float x, float y, float z, float sizeX, float sizeY, float sizeZ )
		{
			glBegin(GL_QUADS);

			glVertex3f(x, y, z);
			glVertex3f(x + sizeX, y, z);
			glVertex3f(x + sizeX, y + sizeY, z);
			glVertex3f(x, y + sizeY, z);

			glVertex3f(x, y, z);
			glVertex3f(x + sizeX, y, z);
			glVertex3f(x + sizeX, y, z + sizeZ);
			glVertex3f(x, y, z + sizeZ);

			glVertex3f(x, y, z);
			glVertex3f(x, y, z + sizeZ);
			glVertex3f(x, y + sizeY, z + sizeZ);
			glVertex3f(x, y + sizeY, z);

			glVertex3f(x + sizeX, y, z);
			glVertex3f(x + sizeX, y, z + sizeZ);
			glVertex3f(x + sizeX, y + sizeY, z + sizeZ);
			glVertex3f(x + sizeX, y + sizeY, z);

			glVertex3f(x, y + sizeY, z);
			glVertex3f(x + sizeX, y + sizeY, z);
			glVertex3f(x + sizeX, y + sizeY, z + sizeZ);
			glVertex3f(x, y + sizeY, z + sizeZ);

			glVertex3f(x + sizeX, y, z + sizeZ);
			glVertex3f(x, y, z + sizeZ);
			glVertex3f(x, y + sizeY, z + sizeZ);
			glVertex3f(x + sizeX, y + sizeY, z + sizeZ);

			glEnd();
		}
		void m4dGUIOGLHapticViewerWidget::DrawCursor(float x, float y, float z, float size)
		{
			glBegin(GL_LINES);
			
			glVertex3f(x, y, z - size / 2.0f);
			glVertex3f(x, y, z + size / 2.0f);

			glVertex3f(x, y - size / 2.0f, z);
			glVertex3f(x, y + size / 2.0f, z);

			glVertex3f(x - size / 2.0f, y, z);
			glVertex3f(x + size / 2.0f, y, z);

			glEnd();
		}
		/*dataGrid M4D::Viewer::m4dGUIOGLHapticViewerWidget::processAverage()
		{
			dataGrid dg;
			
			std::vector< int > row;
			std::vector< std::vector < int > > plane;

			dg.clear();
			for (int i = minX; i < minX + sizeX; i++)
			{	
				std::cout << i << " \n";
				plane.clear();
				for (int j = minY; j < minY + sizeY; j++)
				{
					row.clear();
					for (int k = minZ; k < minZ + sizeZ; k++)
					{
						row.push_back(countAverage(i, j, k));
					}
					plane.push_back(row);
				}
				dg.push_back(plane);
			}
			return dg;
		}*/
		void m4dGUIOGLHapticViewerWidget::medianFilter(int radius)
		{
			dataGrid newOne;
			for (int i = 0; i < myData.size(); i++)
			{
				std::cout << i << std::endl;
				std::vector< std::vector < int > > layer;
				for (int j = 0; j < myData[0].size(); j++)
				{
					std::vector< int > row;
					for (int k = 0; k < myData[0][0].size(); k++)
					{
						std::vector< int > local;
						for (int x = -radius; x <= radius; x++)
						{
							for (int y = -radius; y <= radius; y++)
							{
								if ((((y * y) + (x * x)) <= radius * radius) && ((x + i) >= 0) && ((x + i) < myData.size()) && ((y + j) >= 0) && ((y + j) < myData[i].size()))
								{
									local.push_back(myData[ i + x ][ j + y ][k]);
								}
							}
						}
						sort(local.begin(), local.end());
						row.push_back(local[local.size() / 2]);
					}
					layer.push_back(row);
				}
				newOne.push_back(layer);
			}
			myData = newOne;
		}
		void m4dGUIOGLHapticViewerWidget::preprocessData()
		{

			std::cout << "Preprocessing data...\n";
			loadImageParams();

			std::cout << "Croping data...\n";

			int left[] = {200, 200, 20};
			int right[] = {320, 320, 45};
			std::vector< int > leftCorner(left, left + 3);
			std::vector< int > rightCorner(right, right + 3);
			cropMyImage(leftCorner, rightCorner);
			
			std::cout << "Filtering data...\n";
			medianFilter(2);
			medianFilter(2);
			medianFilter(2);
			
			std::cout << "Making histogram...\n";
			for (int i = minValue; i <= maxValue; i++)
			{
				volumeHistogram.push_back(0);
			}
			
			for (int i = 0; i < myData.size(); i++)
			{
				for (int j = 0; j < myData[0].size(); j++)
				{
					for (int k = 0; k < myData[0][0].size(); k++)
					{
						volumeHistogram[myData[i][j][k]-minValue]++;
					}
				}
			}

			while (volumeHistogram[volumeHistogram.size()-1] == 0)
			{
				maxValue--;
				volumeHistogram.pop_back();
			}
			
			int bordersP[] = {80 , 956, 1000, 1060, 1090, 1131}; // DEBUG
			std::vector< int > borders(bordersP, bordersP+6);
			
			for (int i = 0; i < myData.size(); i++)
			{
				for (int j = 0; j < myData[0].size(); j++)
				{
					for (int k = 0; k < myData[0][0].size(); k++)
					{
						for (int l = 0; l < borders.size(); l++)
						{
							if (myData[i][j][k] < borders[l])
							{
								myData[i][j][k] = l;
								break;
							}
						}
						if (myData[i][j][k] >= borders[borders.size() - 1])
						{
							myData[i][j][k] = borders.size();
						}
					}
				}
			}

			std::cout << "Preparing graphics..." << std::endl;
			prepareGraphics();
		}
		/*int M4D::Viewer::m4dGUIOGLHapticViewerWidget::countAverage(int x, int y, int z)
		{
			int radius = 3; // constant - does not work with another number !!! Must be 3 !! 1 = DEBUG
			int center = 0, wrap1 = 0, wrap2 = 0;
			int centerCount = 0, wrap1Count = 0, wrap2Count = 0;
			int average;

			////DEBUG
			//for ( int i = -radius; i <= radius; i++)
			//{
			//	for (int j = -radius; j <= radius; j++)
			//	{
			//		if ((((x + i) >= minX) && ((x + i) < (minX + sizeX))) && (((y + j) >= minY) && ((y + j) < (minY + sizeY))))
			//		{
			//			centerCount++;
			//			center += imageData[x + i][y + j][z];
			//		}
			//	}
			//}
			//average = center / centerCount;
			
			
			for ( int i = -radius; i <= radius; i++)
			{
				for (int j = -radius; j <= radius; j++)
				{
					if ((((x + i) >= minX) && ((x + i) < (minX + sizeX))) && (((y + j) >= minY) && ((y + j) < (minY + sizeY))))
					{
						if ((i * i + j * j) <= 2)
						{
							centerCount++;
							center += imageData[x + i][y + j][z];
						}
						else if ((i * i + j * j) <= 9)
						{
							wrap1Count++;
							wrap1 += imageData[x + i][y + j][z];
						}
						else
						{
							wrap2Count++;
							wrap2 += imageData[x + i][y + j][z];
						}
					}
				}
			}

			average = ((8 * center / centerCount) + (3 * wrap1 / wrap1Count) + (wrap2 / wrap2Count)) / 12;

			

			return average;
		} */
		void m4dGUIOGLHapticViewerWidget::cropMyImage(const std::vector<int> &firstCorner, const std::vector<int> &secondCorner)
		{
			dataGrid newOne;
			for (int i = MIN(firstCorner[0], secondCorner[0]); i <= MAX(firstCorner[0], secondCorner[0]); i++)
			{
				std::vector< std::vector < int > > layer;
				for (int j = MIN(firstCorner[1], secondCorner[1]); j <= MAX(firstCorner[1], secondCorner[1]); j++)
				{
					std::vector< int > row;
					for (int k = MIN(firstCorner[2], secondCorner[2]); k <= MAX(firstCorner[2], secondCorner[2]); k++)
					{
						NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
							imageID, row.push_back(Imaging::Image< TTYPE, 3 >::CastAImage(inPort->GetDatasetTyped()).GetElement( CreateVector< int32 >(i, j, k) ) ) );
					}
					layer.push_back(row);
				}
				newOne.push_back(layer);
			}
			myData = newOne;
		}
		void m4dGUIOGLHapticViewerWidget::prepareGraphics()
		{
			dataGrid newOne;

			for (int i = 0; i < myData.size(); i++)
			{
				std::vector< std::vector < int > > layer;
				for (int j = 0; j < myData[0].size(); j++)
				{
					std::vector< int > row;
					for (int k = 0; k < myData[0][0].size(); k++)
					{
						row.push_back(-2);
					}
					layer.push_back(row);
				}
				newOne.push_back(layer);
			}

			for (int i = 0; i < myData.size(); i++)
			{
				for (int j = 0; j < myData[0].size(); j++)
				{
					for (int k = 0; k < myData[0][0].size(); k++)
					{
						if (newOne[i][j][k] == -2)
						{
							newOne[i][j][k] = -1;
							floodFillForVolumes(i, j, k, myData[i][j][k], &newOne);
						}
					}
				}
			}

			for (int i = 0; i < myData.size(); i++)
			{
				for (int j = 0; j < myData[0].size(); j++)
				{
					newOne[i][j][0] = myData[i][j][0];
				}
			}
			for (int i = 0; i < myData.size(); i++)
			{
				for (int j = 0; j < myData[0].size(); j++)
				{
					newOne[i][j][newOne[0][0].size() - 1] = myData[i][j][myData[0][0].size() - 1];
				}
			}

			for (int i = 0; i < myData.size(); i++)
			{
				for (int j = 0; j < myData[0][0].size(); j++)
				{
					newOne[i][0][j] = myData[i][0][j];
				}
			}
			for (int i = 0; i < myData.size(); i++)
			{
				for (int j = 0; j < myData[0][0].size(); j++)
				{
					newOne[i][newOne[0][0].size() - 1][j] = myData[i][myData[0][0].size() - 1][j];
				}
			}


			for (int i = 0; i < myData[0].size(); i++)
			{
				for (int j = 0; j < myData[0][0].size(); j++)
				{
					newOne[0][i][j] = myData[0][i][j];
				}
			}
			for (int i = 0; i < myData[0].size(); i++)
			{
				for (int j = 0; j < myData[0][0].size(); j++)
				{
					newOne[newOne[0][0].size() - 1][i][j] = myData[myData[0][0].size() - 1][i][j];
				}
			}
			
			graphicData = newOne;
		}
		void m4dGUIOGLHapticViewerWidget::floodFillForVolumes( int x, int y, int z, int val, dataGrid *newOne )
		{
			std::queue<floodFillDataContainer> myQueue;
			floodFillDataContainer dc(x, y, z, val, newOne);
			myQueue.push(dc);

			while (!myQueue.empty())
			{
				bool continued = false;
				floodFillDataContainer &ffc = myQueue.front();
				for (int i = -1; i <= 1; i++)
				{
					for (int j = -1; j <= 1; j++)
					{
						for (int k = -1; k <= 1; k++)
						{
							if (((ffc.x + i) >= 0) && ((ffc.x + i) < (*newOne).size()) && ((ffc.y + j) >= 0) && ((ffc.y + j) < (*newOne)[0].size()) && ((ffc.z + k) >= 0) && ((ffc.z + k) < (*newOne)[0][0].size()) && ((*newOne)[ffc.x + i][ffc.y + j][ffc.z + k] == -2))
							{
								if (myData[ffc.x + i][ffc.y + j][ffc.z + k] == ffc.val)
								{
									continued = true;
									(*newOne)[ffc.x + i][ffc.y + j][ffc.z + k] = -1;
									floodFillDataContainer ffdc(ffc.x + i, ffc.y + j, ffc.z + k, ffc.val, newOne);
									myQueue.push(ffdc);
								}
							}
						}
					}
				}
				if (!continued)
				{
					(*newOne)[ffc.x][ffc.y][ffc.z] = ffc.val;
				}
				myQueue.pop();
			}
		}
		void m4dGUIOGLHapticViewerWidget::paintGL()
		{
			float offsetWidth = imageSize / 2.0;
			float blockWidth = imageSize / (float)myData.size();

			float blockLength = blockWidth / distanceX * distanceZ;
			float offsetLength = blockLength / 2.0 * (float)myData[0][0].size();

			float colorDif = 1.0 / 7.0;
			
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// Vymaže obrazovku a hloubkový buffer
			glClearColor(0.0, 0.0, 0.0, 0.0);
			glLoadIdentity();

			glTranslatef(0.0f, 0.0f, zoom);
			glRotatef(rotateX, 1.0f, 0.0f, 0.0f);
			glRotatef(rotateY, 0.0f, 1.0f, 0.0f);

			for (int i = 0; i < myData.size(); i++)
			{
				for (int j = 0; j < myData[0].size(); j++)
				{
					for (int k = 0; k < myData[0][0].size(); k++)
					{
						if (graphicData[i][j][k] != -1)
						{
							int val = myData[i][j][k];
							glColor3f(colorDif * (float)val, colorDif * (float)val, colorDif * (float)val);
							DrawBlock(blockWidth * (float)i - offsetWidth, blockWidth * (float)j - offsetWidth, blockLength * (float)k - offsetLength, blockWidth, blockWidth, blockLength);
						}
					}
				}
			}
			
			glLoadIdentity();
			glTranslatef(0.0f, 0.0f, zoom);
			glColor3f(0.0f, 1.0f, 0.0f);
			DrawCursor(cursor->getX()*(imageSize / 2.0), cursor->getY()*(imageSize / 2.0), cursor->getZ()*(imageSize / 2.0), cursorSize);
			
			//glLoadIdentity();
			//glColor3f(1.0f, 0.0f, 0.0f);
			//glTranslatef(0.0f, 0.0f, -5.0f);
			//DrawBlock(0.0f, 0.0f, -5.0f, 1.0f);
			//glLoadIdentity();

			glFlush();

		}
		void m4dGUIOGLHapticViewerWidget::resizeGL(int winW, int winH)
		{
			if (winH == 0)
			{
				winH = 1;
			}

			loadImageParams();
			
			glViewport(0, 0, winW, winH);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			gluPerspective(45.0f,(GLfloat)winW/(GLfloat)winH,0.1f,100.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			updateGL();
		}
		void m4dGUIOGLHapticViewerWidget::loadImageParams()
		{
			imageSize = 10.0;
			cursorSize = imageSize / 20.0;
			imageHeight = imageSize;
			imageWidth = imageSize;
			imageID = inPort->GetDatasetTyped().GetElementTypeID();
			
			if (inPort->GetDatasetTyped().GetDimension() == 3)
			{
				minX = inPort->GetDatasetTyped().GetDimensionExtents(0).minimum;
				minY = inPort->GetDatasetTyped().GetDimensionExtents(1).minimum;
				minZ = inPort->GetDatasetTyped().GetDimensionExtents(2).minimum;

				sizeX = inPort->GetDatasetTyped().GetDimensionExtents(0).maximum - minX;
				sizeY = inPort->GetDatasetTyped().GetDimensionExtents(1).maximum - minY;
				sizeZ = inPort->GetDatasetTyped().GetDimensionExtents(2).maximum - minZ;

				distanceX = inPort->GetDatasetTyped().GetDimensionExtents(0).elementExtent;
				distanceY = inPort->GetDatasetTyped().GetDimensionExtents(1).elementExtent;
				distanceZ = inPort->GetDatasetTyped().GetDimensionExtents(2).elementExtent;
			}

			uint64 min = MAX_INT64;
			uint64 max = 0;
			uint64 result = 0;

			for (int i = minX; i < minX + sizeX; i++)
			{
				for (int j = minY; j < minY + sizeY; j++)
				{
					for (int k = minZ; k < minZ + sizeZ; k++)
					{
						NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
							imageID, result = Imaging::Image< TTYPE, 3 >::CastAImage(inPort->GetDatasetTyped()).GetElement( CreateVector< int32 >(i, j, k) ) );
						if (result > max)
						{
							max = result;
						}
						if (result < min)
						{
							min = result;
						}
					}
				}
			}
			minValue = min;
			maxValue = max;
		}
		void m4dGUIOGLHapticViewerWidget::initializeGL()
		{

			zoom = -imageSize;
			rotateY = 0.0;
			rotateX = 0.0;
			
			glClearColor(0.0f, 0.0f, 0.0f, 0.5f);
			glClearDepth(1.0f);
			glDepthFunc(GL_LEQUAL);
			glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

			//initializeHaptics();
		}

		//void m4dGUIOGLHapticViewerWidget::initializeHaptics()
		//{
		//	buttonStatus = false;
		//	handler = new cHapticDeviceHandler();
		//	/*
		//	// read the number of haptic devices currently connected to the computer
		//	numHapticDevices = handler->getNumDevices();

		//	handler->getDevice(hapticDevice, 0);
		//	hapticDevice->open();
		//	hapticDevice->initialize();
		//	info = hapticDevice->getSpecifications();
		//	*/
		//}

		void m4dGUIOGLHapticViewerWidget::slotSetButtonHandler( ButtonHandler hnd, MouseButton btn )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotSetSelected(bool selected)
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotSetSliceNum( size_t num )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotSetOneSliceMode()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotSetMoreSliceMode( unsigned slicesPerRow, unsigned slicesPerColumn )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotToggleFlipHorizontal()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotToggleFlipVertical()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotAddLeftSideData( std::string data )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotAddRightSideData( std::string data )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotClearLeftSideData()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotClearRightSideData()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotTogglePrintData()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotTogglePrintShapeData()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotZoom( int amount )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotMove( int amountH, int amountV )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotAdjustContrastBrightness( int amountB, int amountC )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotNewPoint( double x, double y, double z )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotNewShape( double x, double y, double z )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotDeletePoint()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotDeleteShape()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotDeleteAll()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotRotateAxisX( double x )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotRotateAxisY( double y )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotRotateAxisZ( double z )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotToggleSliceOrientation()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotColorPicker( double x, double y, double z )
		{

		}
		void m4dGUIOGLHapticViewerWidget::updateViewer()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotMessageHandler( Imaging::PipelineMsgID msgID )
		{
		}
	}
}
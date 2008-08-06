#ifndef DATAPIECE_HEADER_H
#define DATAPIECE_HEADER_H


struct DataPieceHeader
{
  uint32 pieceSize;

  //static void Serialize( DataPieceHeader *h);
  //static void Deserialize( DataPieceHeader *h);

  DataPieceHeader()
    : pieceSize( 0) {}
  DataPieceHeader( uint32 size) 
    : pieceSize(size) {}
};

#endif
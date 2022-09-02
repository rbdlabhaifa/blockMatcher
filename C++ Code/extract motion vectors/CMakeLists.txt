 cmake_minimum_required(VERSION 2.6)
 project(motionVectors)

 set(CMAKE_CXX_STANDARD 11)
 find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h
         HINTS
         $ENV{AVCODECDIR}
         PATH_SUFFIXES include
         PATHS
         ~/Library/Frameworks
         /Library/Frameworks
         /usr/local/include
         /usr/include
         /sw/include
         /opt/local/include
         /opt/csw/include
         /opt/include
         /mingw/include
         )

 #message( "AVUTIL_INCLUDE_DIR is ${AVUTIL_INCLUDE_DIR}" )

 find_library(AVCODEC_LIBRARY
         NAMES avcodec
         HINTS
         $ENV{AVCODECDIR}
         PATH_SUFFIXES lib64 lib bin
         PATHS
         /usr/local
         /usr
         /sw
         /opt/local
         /opt/csw
         /opt
         /mingw
         )
 find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h
         HINTS
         $ENV{AVFORMATDIR}
         PATH_SUFFIXES include
         PATHS
         ~/Library/Frameworks
         /Library/Frameworks
         /usr/local/include
         /usr/include
         /sw/include
         /opt/local/include
         /opt/csw/include
         /opt/include
         /mingw/include
         )

 #message( "AVUTIL_INCLUDE_DIR is ${AVUTIL_INCLUDE_DIR}" )

 find_library(AVFORMAT_LIBRARY
         NAMES avformat
         HINTS
         $ENV{AVFORMATDIR}
         PATH_SUFFIXES lib64 lib bin
         PATHS
         /usr/local
         /usr
         /sw
         /opt/local
         /opt/csw
         /opt
         /mingw
         )
 find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h
         HINTS
         $ENV{AVUTILDIR}
         PATH_SUFFIXES include
         PATHS
         ~/Library/Frameworks
         /Library/Frameworks
         /usr/local/include
         /usr/include
         /sw/include
         /opt/local/include
         /opt/csw/include
         /opt/include
         /mingw/include
         )

 #message( "AVUTIL_INCLUDE_DIR is ${AVUTIL_INCLUDE_DIR}" )

 find_library(AVUTIL_LIBRARY
         NAMES avutil
         HINTS
         $ENV{AVUTILDIR}
         PATH_SUFFIXES lib64 lib bin
         PATHS
         /usr/local
         /usr
         /sw
         /opt/local
         /opt/csw
         /opt
         /mingw
         )
 add_executable(motionVectors test.cpp)
 target_include_directories(motionVectors PRIVATE ${AVCODEC_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR})
 target_link_libraries(motionVectors ${AVCODEC_LIBRARY} ${AVUTIL_LIBRARY} ${AVFORMAT_LIBRARY})
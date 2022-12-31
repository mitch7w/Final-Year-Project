# Install script for directory: /Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/bin/freenect-camtest")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-camtest" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-camtest")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/lib/libfreenect.0.dylib" "libfreenect.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-camtest")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-camtest")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/bin/freenect-wavrecord")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-wavrecord" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-wavrecord")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/lib/libfreenect.0.dylib" "libfreenect.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-wavrecord")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-wavrecord")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/bin/freenect-glview")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-glview" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-glview")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/lib/libfreenect.0.dylib" "libfreenect.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-glview")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-glview")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/bin/freenect-regview")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-regview" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-regview")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/lib/libfreenect.0.dylib" "libfreenect.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-regview")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-regview")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/bin/freenect-hiview")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-hiview" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-hiview")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/lib/libfreenect.0.dylib" "libfreenect.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-hiview")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-hiview")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/bin/freenect-chunkview")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-chunkview" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-chunkview")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/lib/libfreenect.0.dylib" "libfreenect.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-chunkview")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-chunkview")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/bin/freenect-micview")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-micview" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-micview")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/lib/libfreenect.0.dylib" "libfreenect.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-micview")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-micview")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/bin/freenect-regtest")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-regtest" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-regtest")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/lib/libfreenect.0.dylib" "libfreenect.0.dylib"
      -change "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/lib/libfreenect_sync.0.dylib" "libfreenect_sync.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-regtest")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-regtest")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/bin/freenect-tiltdemo")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-tiltdemo" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-tiltdemo")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/lib/libfreenect.0.dylib" "libfreenect.0.dylib"
      -change "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/lib/libfreenect_sync.0.dylib" "libfreenect_sync.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-tiltdemo")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-tiltdemo")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/bin/freenect-glpclview")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-glpclview" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-glpclview")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/lib/libfreenect.0.dylib" "libfreenect.0.dylib"
      -change "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/lib/libfreenect_sync.0.dylib" "libfreenect_sync.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-glpclview")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/freenect-glpclview")
    endif()
  endif()
endif()


# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.23.0/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.23.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build"

# Include any dependencies generated for this target.
include wrappers/cpp/CMakeFiles/freenect-cppview.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include wrappers/cpp/CMakeFiles/freenect-cppview.dir/compiler_depend.make

# Include the progress variables for this target.
include wrappers/cpp/CMakeFiles/freenect-cppview.dir/progress.make

# Include the compile flags for this target's objects.
include wrappers/cpp/CMakeFiles/freenect-cppview.dir/flags.make

wrappers/cpp/CMakeFiles/freenect-cppview.dir/cppview.cpp.o: wrappers/cpp/CMakeFiles/freenect-cppview.dir/flags.make
wrappers/cpp/CMakeFiles/freenect-cppview.dir/cppview.cpp.o: ../wrappers/cpp/cppview.cpp
wrappers/cpp/CMakeFiles/freenect-cppview.dir/cppview.cpp.o: wrappers/cpp/CMakeFiles/freenect-cppview.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object wrappers/cpp/CMakeFiles/freenect-cppview.dir/cppview.cpp.o"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/cpp" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT wrappers/cpp/CMakeFiles/freenect-cppview.dir/cppview.cpp.o -MF CMakeFiles/freenect-cppview.dir/cppview.cpp.o.d -o CMakeFiles/freenect-cppview.dir/cppview.cpp.o -c "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/cpp/cppview.cpp"

wrappers/cpp/CMakeFiles/freenect-cppview.dir/cppview.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/freenect-cppview.dir/cppview.cpp.i"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/cpp" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/cpp/cppview.cpp" > CMakeFiles/freenect-cppview.dir/cppview.cpp.i

wrappers/cpp/CMakeFiles/freenect-cppview.dir/cppview.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/freenect-cppview.dir/cppview.cpp.s"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/cpp" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/cpp/cppview.cpp" -o CMakeFiles/freenect-cppview.dir/cppview.cpp.s

# Object files for target freenect-cppview
freenect__cppview_OBJECTS = \
"CMakeFiles/freenect-cppview.dir/cppview.cpp.o"

# External object files for target freenect-cppview
freenect__cppview_EXTERNAL_OBJECTS =

bin/freenect-cppview: wrappers/cpp/CMakeFiles/freenect-cppview.dir/cppview.cpp.o
bin/freenect-cppview: wrappers/cpp/CMakeFiles/freenect-cppview.dir/build.make
bin/freenect-cppview: lib/libfreenect.0.6.3.dylib
bin/freenect-cppview: /opt/homebrew/lib/libusb-1.0.dylib
bin/freenect-cppview: wrappers/cpp/CMakeFiles/freenect-cppview.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/freenect-cppview"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/cpp" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/freenect-cppview.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
wrappers/cpp/CMakeFiles/freenect-cppview.dir/build: bin/freenect-cppview
.PHONY : wrappers/cpp/CMakeFiles/freenect-cppview.dir/build

wrappers/cpp/CMakeFiles/freenect-cppview.dir/clean:
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/cpp" && $(CMAKE_COMMAND) -P CMakeFiles/freenect-cppview.dir/cmake_clean.cmake
.PHONY : wrappers/cpp/CMakeFiles/freenect-cppview.dir/clean

wrappers/cpp/CMakeFiles/freenect-cppview.dir/depend:
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/cpp" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/cpp" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/cpp/CMakeFiles/freenect-cppview.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : wrappers/cpp/CMakeFiles/freenect-cppview.dir/depend


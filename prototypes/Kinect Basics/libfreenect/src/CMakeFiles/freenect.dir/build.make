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
CMAKE_BINARY_DIR = "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect"

# Include any dependencies generated for this target.
include src/CMakeFiles/freenect.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/freenect.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/freenect.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/freenect.dir/flags.make

src/CMakeFiles/freenect.dir/core.c.o: src/CMakeFiles/freenect.dir/flags.make
src/CMakeFiles/freenect.dir/core.c.o: src/core.c
src/CMakeFiles/freenect.dir/core.c.o: src/CMakeFiles/freenect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/CMakeFiles/freenect.dir/core.c.o"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/freenect.dir/core.c.o -MF CMakeFiles/freenect.dir/core.c.o.d -o CMakeFiles/freenect.dir/core.c.o -c "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/core.c"

src/CMakeFiles/freenect.dir/core.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freenect.dir/core.c.i"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/core.c" > CMakeFiles/freenect.dir/core.c.i

src/CMakeFiles/freenect.dir/core.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freenect.dir/core.c.s"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/core.c" -o CMakeFiles/freenect.dir/core.c.s

src/CMakeFiles/freenect.dir/tilt.c.o: src/CMakeFiles/freenect.dir/flags.make
src/CMakeFiles/freenect.dir/tilt.c.o: src/tilt.c
src/CMakeFiles/freenect.dir/tilt.c.o: src/CMakeFiles/freenect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building C object src/CMakeFiles/freenect.dir/tilt.c.o"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/freenect.dir/tilt.c.o -MF CMakeFiles/freenect.dir/tilt.c.o.d -o CMakeFiles/freenect.dir/tilt.c.o -c "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/tilt.c"

src/CMakeFiles/freenect.dir/tilt.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freenect.dir/tilt.c.i"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/tilt.c" > CMakeFiles/freenect.dir/tilt.c.i

src/CMakeFiles/freenect.dir/tilt.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freenect.dir/tilt.c.s"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/tilt.c" -o CMakeFiles/freenect.dir/tilt.c.s

src/CMakeFiles/freenect.dir/cameras.c.o: src/CMakeFiles/freenect.dir/flags.make
src/CMakeFiles/freenect.dir/cameras.c.o: src/cameras.c
src/CMakeFiles/freenect.dir/cameras.c.o: src/CMakeFiles/freenect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building C object src/CMakeFiles/freenect.dir/cameras.c.o"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/freenect.dir/cameras.c.o -MF CMakeFiles/freenect.dir/cameras.c.o.d -o CMakeFiles/freenect.dir/cameras.c.o -c "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/cameras.c"

src/CMakeFiles/freenect.dir/cameras.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freenect.dir/cameras.c.i"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/cameras.c" > CMakeFiles/freenect.dir/cameras.c.i

src/CMakeFiles/freenect.dir/cameras.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freenect.dir/cameras.c.s"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/cameras.c" -o CMakeFiles/freenect.dir/cameras.c.s

src/CMakeFiles/freenect.dir/flags.c.o: src/CMakeFiles/freenect.dir/flags.make
src/CMakeFiles/freenect.dir/flags.c.o: src/flags.c
src/CMakeFiles/freenect.dir/flags.c.o: src/CMakeFiles/freenect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building C object src/CMakeFiles/freenect.dir/flags.c.o"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/freenect.dir/flags.c.o -MF CMakeFiles/freenect.dir/flags.c.o.d -o CMakeFiles/freenect.dir/flags.c.o -c "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/flags.c"

src/CMakeFiles/freenect.dir/flags.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freenect.dir/flags.c.i"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/flags.c" > CMakeFiles/freenect.dir/flags.c.i

src/CMakeFiles/freenect.dir/flags.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freenect.dir/flags.c.s"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/flags.c" -o CMakeFiles/freenect.dir/flags.c.s

src/CMakeFiles/freenect.dir/usb_libusb10.c.o: src/CMakeFiles/freenect.dir/flags.make
src/CMakeFiles/freenect.dir/usb_libusb10.c.o: src/usb_libusb10.c
src/CMakeFiles/freenect.dir/usb_libusb10.c.o: src/CMakeFiles/freenect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Building C object src/CMakeFiles/freenect.dir/usb_libusb10.c.o"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/freenect.dir/usb_libusb10.c.o -MF CMakeFiles/freenect.dir/usb_libusb10.c.o.d -o CMakeFiles/freenect.dir/usb_libusb10.c.o -c "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/usb_libusb10.c"

src/CMakeFiles/freenect.dir/usb_libusb10.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freenect.dir/usb_libusb10.c.i"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/usb_libusb10.c" > CMakeFiles/freenect.dir/usb_libusb10.c.i

src/CMakeFiles/freenect.dir/usb_libusb10.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freenect.dir/usb_libusb10.c.s"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/usb_libusb10.c" -o CMakeFiles/freenect.dir/usb_libusb10.c.s

src/CMakeFiles/freenect.dir/registration.c.o: src/CMakeFiles/freenect.dir/flags.make
src/CMakeFiles/freenect.dir/registration.c.o: src/registration.c
src/CMakeFiles/freenect.dir/registration.c.o: src/CMakeFiles/freenect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_6) "Building C object src/CMakeFiles/freenect.dir/registration.c.o"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/freenect.dir/registration.c.o -MF CMakeFiles/freenect.dir/registration.c.o.d -o CMakeFiles/freenect.dir/registration.c.o -c "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/registration.c"

src/CMakeFiles/freenect.dir/registration.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freenect.dir/registration.c.i"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/registration.c" > CMakeFiles/freenect.dir/registration.c.i

src/CMakeFiles/freenect.dir/registration.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freenect.dir/registration.c.s"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/registration.c" -o CMakeFiles/freenect.dir/registration.c.s

src/CMakeFiles/freenect.dir/audio.c.o: src/CMakeFiles/freenect.dir/flags.make
src/CMakeFiles/freenect.dir/audio.c.o: src/audio.c
src/CMakeFiles/freenect.dir/audio.c.o: src/CMakeFiles/freenect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_7) "Building C object src/CMakeFiles/freenect.dir/audio.c.o"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/freenect.dir/audio.c.o -MF CMakeFiles/freenect.dir/audio.c.o.d -o CMakeFiles/freenect.dir/audio.c.o -c "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/audio.c"

src/CMakeFiles/freenect.dir/audio.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freenect.dir/audio.c.i"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/audio.c" > CMakeFiles/freenect.dir/audio.c.i

src/CMakeFiles/freenect.dir/audio.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freenect.dir/audio.c.s"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/audio.c" -o CMakeFiles/freenect.dir/audio.c.s

src/CMakeFiles/freenect.dir/loader.c.o: src/CMakeFiles/freenect.dir/flags.make
src/CMakeFiles/freenect.dir/loader.c.o: src/loader.c
src/CMakeFiles/freenect.dir/loader.c.o: src/CMakeFiles/freenect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_8) "Building C object src/CMakeFiles/freenect.dir/loader.c.o"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/freenect.dir/loader.c.o -MF CMakeFiles/freenect.dir/loader.c.o.d -o CMakeFiles/freenect.dir/loader.c.o -c "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/loader.c"

src/CMakeFiles/freenect.dir/loader.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freenect.dir/loader.c.i"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/loader.c" > CMakeFiles/freenect.dir/loader.c.i

src/CMakeFiles/freenect.dir/loader.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freenect.dir/loader.c.s"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/loader.c" -o CMakeFiles/freenect.dir/loader.c.s

# Object files for target freenect
freenect_OBJECTS = \
"CMakeFiles/freenect.dir/core.c.o" \
"CMakeFiles/freenect.dir/tilt.c.o" \
"CMakeFiles/freenect.dir/cameras.c.o" \
"CMakeFiles/freenect.dir/flags.c.o" \
"CMakeFiles/freenect.dir/usb_libusb10.c.o" \
"CMakeFiles/freenect.dir/registration.c.o" \
"CMakeFiles/freenect.dir/audio.c.o" \
"CMakeFiles/freenect.dir/loader.c.o"

# External object files for target freenect
freenect_EXTERNAL_OBJECTS =

lib/libfreenect.0.6.3.dylib: src/CMakeFiles/freenect.dir/core.c.o
lib/libfreenect.0.6.3.dylib: src/CMakeFiles/freenect.dir/tilt.c.o
lib/libfreenect.0.6.3.dylib: src/CMakeFiles/freenect.dir/cameras.c.o
lib/libfreenect.0.6.3.dylib: src/CMakeFiles/freenect.dir/flags.c.o
lib/libfreenect.0.6.3.dylib: src/CMakeFiles/freenect.dir/usb_libusb10.c.o
lib/libfreenect.0.6.3.dylib: src/CMakeFiles/freenect.dir/registration.c.o
lib/libfreenect.0.6.3.dylib: src/CMakeFiles/freenect.dir/audio.c.o
lib/libfreenect.0.6.3.dylib: src/CMakeFiles/freenect.dir/loader.c.o
lib/libfreenect.0.6.3.dylib: src/CMakeFiles/freenect.dir/build.make
lib/libfreenect.0.6.3.dylib: /opt/homebrew/lib/libusb-1.0.dylib
lib/libfreenect.0.6.3.dylib: src/CMakeFiles/freenect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_9) "Linking C shared library ../lib/libfreenect.dylib"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/freenect.dir/link.txt --verbose=$(VERBOSE)
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && $(CMAKE_COMMAND) -E cmake_symlink_library ../lib/libfreenect.0.6.3.dylib ../lib/libfreenect.0.dylib ../lib/libfreenect.dylib

lib/libfreenect.0.dylib: lib/libfreenect.0.6.3.dylib
	@$(CMAKE_COMMAND) -E touch_nocreate lib/libfreenect.0.dylib

lib/libfreenect.dylib: lib/libfreenect.0.6.3.dylib
	@$(CMAKE_COMMAND) -E touch_nocreate lib/libfreenect.dylib

# Rule to build all files generated by this target.
src/CMakeFiles/freenect.dir/build: lib/libfreenect.dylib
.PHONY : src/CMakeFiles/freenect.dir/build

src/CMakeFiles/freenect.dir/clean:
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" && $(CMAKE_COMMAND) -P CMakeFiles/freenect.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/freenect.dir/clean

src/CMakeFiles/freenect.dir/depend:
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/src/CMakeFiles/freenect.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : src/CMakeFiles/freenect.dir/depend


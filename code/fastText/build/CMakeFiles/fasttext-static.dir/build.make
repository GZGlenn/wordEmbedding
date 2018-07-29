# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/public/code/wordEmbedding/code/fastText

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/public/code/wordEmbedding/code/fastText/build

# Include any dependencies generated for this target.
include CMakeFiles/fasttext-static.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fasttext-static.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fasttext-static.dir/flags.make

CMakeFiles/fasttext-static.dir/src/args.cc.o: CMakeFiles/fasttext-static.dir/flags.make
CMakeFiles/fasttext-static.dir/src/args.cc.o: ../src/args.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/public/code/wordEmbedding/code/fastText/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fasttext-static.dir/src/args.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fasttext-static.dir/src/args.cc.o -c /home/public/code/wordEmbedding/code/fastText/src/args.cc

CMakeFiles/fasttext-static.dir/src/args.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fasttext-static.dir/src/args.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/public/code/wordEmbedding/code/fastText/src/args.cc > CMakeFiles/fasttext-static.dir/src/args.cc.i

CMakeFiles/fasttext-static.dir/src/args.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fasttext-static.dir/src/args.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/public/code/wordEmbedding/code/fastText/src/args.cc -o CMakeFiles/fasttext-static.dir/src/args.cc.s

CMakeFiles/fasttext-static.dir/src/args.cc.o.requires:

.PHONY : CMakeFiles/fasttext-static.dir/src/args.cc.o.requires

CMakeFiles/fasttext-static.dir/src/args.cc.o.provides: CMakeFiles/fasttext-static.dir/src/args.cc.o.requires
	$(MAKE) -f CMakeFiles/fasttext-static.dir/build.make CMakeFiles/fasttext-static.dir/src/args.cc.o.provides.build
.PHONY : CMakeFiles/fasttext-static.dir/src/args.cc.o.provides

CMakeFiles/fasttext-static.dir/src/args.cc.o.provides.build: CMakeFiles/fasttext-static.dir/src/args.cc.o


CMakeFiles/fasttext-static.dir/src/dictionary.cc.o: CMakeFiles/fasttext-static.dir/flags.make
CMakeFiles/fasttext-static.dir/src/dictionary.cc.o: ../src/dictionary.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/public/code/wordEmbedding/code/fastText/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/fasttext-static.dir/src/dictionary.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fasttext-static.dir/src/dictionary.cc.o -c /home/public/code/wordEmbedding/code/fastText/src/dictionary.cc

CMakeFiles/fasttext-static.dir/src/dictionary.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fasttext-static.dir/src/dictionary.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/public/code/wordEmbedding/code/fastText/src/dictionary.cc > CMakeFiles/fasttext-static.dir/src/dictionary.cc.i

CMakeFiles/fasttext-static.dir/src/dictionary.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fasttext-static.dir/src/dictionary.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/public/code/wordEmbedding/code/fastText/src/dictionary.cc -o CMakeFiles/fasttext-static.dir/src/dictionary.cc.s

CMakeFiles/fasttext-static.dir/src/dictionary.cc.o.requires:

.PHONY : CMakeFiles/fasttext-static.dir/src/dictionary.cc.o.requires

CMakeFiles/fasttext-static.dir/src/dictionary.cc.o.provides: CMakeFiles/fasttext-static.dir/src/dictionary.cc.o.requires
	$(MAKE) -f CMakeFiles/fasttext-static.dir/build.make CMakeFiles/fasttext-static.dir/src/dictionary.cc.o.provides.build
.PHONY : CMakeFiles/fasttext-static.dir/src/dictionary.cc.o.provides

CMakeFiles/fasttext-static.dir/src/dictionary.cc.o.provides.build: CMakeFiles/fasttext-static.dir/src/dictionary.cc.o


CMakeFiles/fasttext-static.dir/src/fasttext.cc.o: CMakeFiles/fasttext-static.dir/flags.make
CMakeFiles/fasttext-static.dir/src/fasttext.cc.o: ../src/fasttext.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/public/code/wordEmbedding/code/fastText/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/fasttext-static.dir/src/fasttext.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fasttext-static.dir/src/fasttext.cc.o -c /home/public/code/wordEmbedding/code/fastText/src/fasttext.cc

CMakeFiles/fasttext-static.dir/src/fasttext.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fasttext-static.dir/src/fasttext.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/public/code/wordEmbedding/code/fastText/src/fasttext.cc > CMakeFiles/fasttext-static.dir/src/fasttext.cc.i

CMakeFiles/fasttext-static.dir/src/fasttext.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fasttext-static.dir/src/fasttext.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/public/code/wordEmbedding/code/fastText/src/fasttext.cc -o CMakeFiles/fasttext-static.dir/src/fasttext.cc.s

CMakeFiles/fasttext-static.dir/src/fasttext.cc.o.requires:

.PHONY : CMakeFiles/fasttext-static.dir/src/fasttext.cc.o.requires

CMakeFiles/fasttext-static.dir/src/fasttext.cc.o.provides: CMakeFiles/fasttext-static.dir/src/fasttext.cc.o.requires
	$(MAKE) -f CMakeFiles/fasttext-static.dir/build.make CMakeFiles/fasttext-static.dir/src/fasttext.cc.o.provides.build
.PHONY : CMakeFiles/fasttext-static.dir/src/fasttext.cc.o.provides

CMakeFiles/fasttext-static.dir/src/fasttext.cc.o.provides.build: CMakeFiles/fasttext-static.dir/src/fasttext.cc.o


CMakeFiles/fasttext-static.dir/src/main.cc.o: CMakeFiles/fasttext-static.dir/flags.make
CMakeFiles/fasttext-static.dir/src/main.cc.o: ../src/main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/public/code/wordEmbedding/code/fastText/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/fasttext-static.dir/src/main.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fasttext-static.dir/src/main.cc.o -c /home/public/code/wordEmbedding/code/fastText/src/main.cc

CMakeFiles/fasttext-static.dir/src/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fasttext-static.dir/src/main.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/public/code/wordEmbedding/code/fastText/src/main.cc > CMakeFiles/fasttext-static.dir/src/main.cc.i

CMakeFiles/fasttext-static.dir/src/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fasttext-static.dir/src/main.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/public/code/wordEmbedding/code/fastText/src/main.cc -o CMakeFiles/fasttext-static.dir/src/main.cc.s

CMakeFiles/fasttext-static.dir/src/main.cc.o.requires:

.PHONY : CMakeFiles/fasttext-static.dir/src/main.cc.o.requires

CMakeFiles/fasttext-static.dir/src/main.cc.o.provides: CMakeFiles/fasttext-static.dir/src/main.cc.o.requires
	$(MAKE) -f CMakeFiles/fasttext-static.dir/build.make CMakeFiles/fasttext-static.dir/src/main.cc.o.provides.build
.PHONY : CMakeFiles/fasttext-static.dir/src/main.cc.o.provides

CMakeFiles/fasttext-static.dir/src/main.cc.o.provides.build: CMakeFiles/fasttext-static.dir/src/main.cc.o


CMakeFiles/fasttext-static.dir/src/matrix.cc.o: CMakeFiles/fasttext-static.dir/flags.make
CMakeFiles/fasttext-static.dir/src/matrix.cc.o: ../src/matrix.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/public/code/wordEmbedding/code/fastText/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/fasttext-static.dir/src/matrix.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fasttext-static.dir/src/matrix.cc.o -c /home/public/code/wordEmbedding/code/fastText/src/matrix.cc

CMakeFiles/fasttext-static.dir/src/matrix.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fasttext-static.dir/src/matrix.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/public/code/wordEmbedding/code/fastText/src/matrix.cc > CMakeFiles/fasttext-static.dir/src/matrix.cc.i

CMakeFiles/fasttext-static.dir/src/matrix.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fasttext-static.dir/src/matrix.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/public/code/wordEmbedding/code/fastText/src/matrix.cc -o CMakeFiles/fasttext-static.dir/src/matrix.cc.s

CMakeFiles/fasttext-static.dir/src/matrix.cc.o.requires:

.PHONY : CMakeFiles/fasttext-static.dir/src/matrix.cc.o.requires

CMakeFiles/fasttext-static.dir/src/matrix.cc.o.provides: CMakeFiles/fasttext-static.dir/src/matrix.cc.o.requires
	$(MAKE) -f CMakeFiles/fasttext-static.dir/build.make CMakeFiles/fasttext-static.dir/src/matrix.cc.o.provides.build
.PHONY : CMakeFiles/fasttext-static.dir/src/matrix.cc.o.provides

CMakeFiles/fasttext-static.dir/src/matrix.cc.o.provides.build: CMakeFiles/fasttext-static.dir/src/matrix.cc.o


CMakeFiles/fasttext-static.dir/src/model.cc.o: CMakeFiles/fasttext-static.dir/flags.make
CMakeFiles/fasttext-static.dir/src/model.cc.o: ../src/model.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/public/code/wordEmbedding/code/fastText/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/fasttext-static.dir/src/model.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fasttext-static.dir/src/model.cc.o -c /home/public/code/wordEmbedding/code/fastText/src/model.cc

CMakeFiles/fasttext-static.dir/src/model.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fasttext-static.dir/src/model.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/public/code/wordEmbedding/code/fastText/src/model.cc > CMakeFiles/fasttext-static.dir/src/model.cc.i

CMakeFiles/fasttext-static.dir/src/model.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fasttext-static.dir/src/model.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/public/code/wordEmbedding/code/fastText/src/model.cc -o CMakeFiles/fasttext-static.dir/src/model.cc.s

CMakeFiles/fasttext-static.dir/src/model.cc.o.requires:

.PHONY : CMakeFiles/fasttext-static.dir/src/model.cc.o.requires

CMakeFiles/fasttext-static.dir/src/model.cc.o.provides: CMakeFiles/fasttext-static.dir/src/model.cc.o.requires
	$(MAKE) -f CMakeFiles/fasttext-static.dir/build.make CMakeFiles/fasttext-static.dir/src/model.cc.o.provides.build
.PHONY : CMakeFiles/fasttext-static.dir/src/model.cc.o.provides

CMakeFiles/fasttext-static.dir/src/model.cc.o.provides.build: CMakeFiles/fasttext-static.dir/src/model.cc.o


CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o: CMakeFiles/fasttext-static.dir/flags.make
CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o: ../src/productquantizer.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/public/code/wordEmbedding/code/fastText/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o -c /home/public/code/wordEmbedding/code/fastText/src/productquantizer.cc

CMakeFiles/fasttext-static.dir/src/productquantizer.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fasttext-static.dir/src/productquantizer.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/public/code/wordEmbedding/code/fastText/src/productquantizer.cc > CMakeFiles/fasttext-static.dir/src/productquantizer.cc.i

CMakeFiles/fasttext-static.dir/src/productquantizer.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fasttext-static.dir/src/productquantizer.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/public/code/wordEmbedding/code/fastText/src/productquantizer.cc -o CMakeFiles/fasttext-static.dir/src/productquantizer.cc.s

CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o.requires:

.PHONY : CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o.requires

CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o.provides: CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o.requires
	$(MAKE) -f CMakeFiles/fasttext-static.dir/build.make CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o.provides.build
.PHONY : CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o.provides

CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o.provides.build: CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o


CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o: CMakeFiles/fasttext-static.dir/flags.make
CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o: ../src/qmatrix.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/public/code/wordEmbedding/code/fastText/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o -c /home/public/code/wordEmbedding/code/fastText/src/qmatrix.cc

CMakeFiles/fasttext-static.dir/src/qmatrix.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fasttext-static.dir/src/qmatrix.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/public/code/wordEmbedding/code/fastText/src/qmatrix.cc > CMakeFiles/fasttext-static.dir/src/qmatrix.cc.i

CMakeFiles/fasttext-static.dir/src/qmatrix.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fasttext-static.dir/src/qmatrix.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/public/code/wordEmbedding/code/fastText/src/qmatrix.cc -o CMakeFiles/fasttext-static.dir/src/qmatrix.cc.s

CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o.requires:

.PHONY : CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o.requires

CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o.provides: CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o.requires
	$(MAKE) -f CMakeFiles/fasttext-static.dir/build.make CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o.provides.build
.PHONY : CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o.provides

CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o.provides.build: CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o


CMakeFiles/fasttext-static.dir/src/utils.cc.o: CMakeFiles/fasttext-static.dir/flags.make
CMakeFiles/fasttext-static.dir/src/utils.cc.o: ../src/utils.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/public/code/wordEmbedding/code/fastText/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/fasttext-static.dir/src/utils.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fasttext-static.dir/src/utils.cc.o -c /home/public/code/wordEmbedding/code/fastText/src/utils.cc

CMakeFiles/fasttext-static.dir/src/utils.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fasttext-static.dir/src/utils.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/public/code/wordEmbedding/code/fastText/src/utils.cc > CMakeFiles/fasttext-static.dir/src/utils.cc.i

CMakeFiles/fasttext-static.dir/src/utils.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fasttext-static.dir/src/utils.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/public/code/wordEmbedding/code/fastText/src/utils.cc -o CMakeFiles/fasttext-static.dir/src/utils.cc.s

CMakeFiles/fasttext-static.dir/src/utils.cc.o.requires:

.PHONY : CMakeFiles/fasttext-static.dir/src/utils.cc.o.requires

CMakeFiles/fasttext-static.dir/src/utils.cc.o.provides: CMakeFiles/fasttext-static.dir/src/utils.cc.o.requires
	$(MAKE) -f CMakeFiles/fasttext-static.dir/build.make CMakeFiles/fasttext-static.dir/src/utils.cc.o.provides.build
.PHONY : CMakeFiles/fasttext-static.dir/src/utils.cc.o.provides

CMakeFiles/fasttext-static.dir/src/utils.cc.o.provides.build: CMakeFiles/fasttext-static.dir/src/utils.cc.o


CMakeFiles/fasttext-static.dir/src/vector.cc.o: CMakeFiles/fasttext-static.dir/flags.make
CMakeFiles/fasttext-static.dir/src/vector.cc.o: ../src/vector.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/public/code/wordEmbedding/code/fastText/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/fasttext-static.dir/src/vector.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fasttext-static.dir/src/vector.cc.o -c /home/public/code/wordEmbedding/code/fastText/src/vector.cc

CMakeFiles/fasttext-static.dir/src/vector.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fasttext-static.dir/src/vector.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/public/code/wordEmbedding/code/fastText/src/vector.cc > CMakeFiles/fasttext-static.dir/src/vector.cc.i

CMakeFiles/fasttext-static.dir/src/vector.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fasttext-static.dir/src/vector.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/public/code/wordEmbedding/code/fastText/src/vector.cc -o CMakeFiles/fasttext-static.dir/src/vector.cc.s

CMakeFiles/fasttext-static.dir/src/vector.cc.o.requires:

.PHONY : CMakeFiles/fasttext-static.dir/src/vector.cc.o.requires

CMakeFiles/fasttext-static.dir/src/vector.cc.o.provides: CMakeFiles/fasttext-static.dir/src/vector.cc.o.requires
	$(MAKE) -f CMakeFiles/fasttext-static.dir/build.make CMakeFiles/fasttext-static.dir/src/vector.cc.o.provides.build
.PHONY : CMakeFiles/fasttext-static.dir/src/vector.cc.o.provides

CMakeFiles/fasttext-static.dir/src/vector.cc.o.provides.build: CMakeFiles/fasttext-static.dir/src/vector.cc.o


# Object files for target fasttext-static
fasttext__static_OBJECTS = \
"CMakeFiles/fasttext-static.dir/src/args.cc.o" \
"CMakeFiles/fasttext-static.dir/src/dictionary.cc.o" \
"CMakeFiles/fasttext-static.dir/src/fasttext.cc.o" \
"CMakeFiles/fasttext-static.dir/src/main.cc.o" \
"CMakeFiles/fasttext-static.dir/src/matrix.cc.o" \
"CMakeFiles/fasttext-static.dir/src/model.cc.o" \
"CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o" \
"CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o" \
"CMakeFiles/fasttext-static.dir/src/utils.cc.o" \
"CMakeFiles/fasttext-static.dir/src/vector.cc.o"

# External object files for target fasttext-static
fasttext__static_EXTERNAL_OBJECTS =

libfasttext.a: CMakeFiles/fasttext-static.dir/src/args.cc.o
libfasttext.a: CMakeFiles/fasttext-static.dir/src/dictionary.cc.o
libfasttext.a: CMakeFiles/fasttext-static.dir/src/fasttext.cc.o
libfasttext.a: CMakeFiles/fasttext-static.dir/src/main.cc.o
libfasttext.a: CMakeFiles/fasttext-static.dir/src/matrix.cc.o
libfasttext.a: CMakeFiles/fasttext-static.dir/src/model.cc.o
libfasttext.a: CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o
libfasttext.a: CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o
libfasttext.a: CMakeFiles/fasttext-static.dir/src/utils.cc.o
libfasttext.a: CMakeFiles/fasttext-static.dir/src/vector.cc.o
libfasttext.a: CMakeFiles/fasttext-static.dir/build.make
libfasttext.a: CMakeFiles/fasttext-static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/public/code/wordEmbedding/code/fastText/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX static library libfasttext.a"
	$(CMAKE_COMMAND) -P CMakeFiles/fasttext-static.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fasttext-static.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fasttext-static.dir/build: libfasttext.a

.PHONY : CMakeFiles/fasttext-static.dir/build

CMakeFiles/fasttext-static.dir/requires: CMakeFiles/fasttext-static.dir/src/args.cc.o.requires
CMakeFiles/fasttext-static.dir/requires: CMakeFiles/fasttext-static.dir/src/dictionary.cc.o.requires
CMakeFiles/fasttext-static.dir/requires: CMakeFiles/fasttext-static.dir/src/fasttext.cc.o.requires
CMakeFiles/fasttext-static.dir/requires: CMakeFiles/fasttext-static.dir/src/main.cc.o.requires
CMakeFiles/fasttext-static.dir/requires: CMakeFiles/fasttext-static.dir/src/matrix.cc.o.requires
CMakeFiles/fasttext-static.dir/requires: CMakeFiles/fasttext-static.dir/src/model.cc.o.requires
CMakeFiles/fasttext-static.dir/requires: CMakeFiles/fasttext-static.dir/src/productquantizer.cc.o.requires
CMakeFiles/fasttext-static.dir/requires: CMakeFiles/fasttext-static.dir/src/qmatrix.cc.o.requires
CMakeFiles/fasttext-static.dir/requires: CMakeFiles/fasttext-static.dir/src/utils.cc.o.requires
CMakeFiles/fasttext-static.dir/requires: CMakeFiles/fasttext-static.dir/src/vector.cc.o.requires

.PHONY : CMakeFiles/fasttext-static.dir/requires

CMakeFiles/fasttext-static.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fasttext-static.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fasttext-static.dir/clean

CMakeFiles/fasttext-static.dir/depend:
	cd /home/public/code/wordEmbedding/code/fastText/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/public/code/wordEmbedding/code/fastText /home/public/code/wordEmbedding/code/fastText /home/public/code/wordEmbedding/code/fastText/build /home/public/code/wordEmbedding/code/fastText/build /home/public/code/wordEmbedding/code/fastText/build/CMakeFiles/fasttext-static.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fasttext-static.dir/depend


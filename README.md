# Extracting Handwriting from Lecture Videos
The aim of this project is to extract handwritng on a whiteboard or a chalkboard from a lecture video.
The program first extracts a background color and then uses OpenCV morphilogical transformations to extract handwritten lines, curves and texts.
If a part of handwriting is temperally occluded by people in front of the board, it can be kept in the extraction results.

## Installation
### Prerequisites
* C++11 compiler
* CMake 3.11+
* OpenCV (tested with 4.5.1)
* Boost (tested with 1.66)

### Build Process
Clone the project.
```
$ git clone https://github.com/pochun/HandwritingExtraction.git
```

Then create build directory

```
$ cd HandwritingExtraction
$ mkdir build
```

and generate Makefile in the build directory, and build the project.

```
$ cd build
$ cmake ..
$ make
```
## Usage
### Options
```
  -h [ --help ]         Help screen
  -i [ --input ] arg    Input video
  -o [ --output ] arg   Output video
```
### Example
```
./handwriting_extraction -i test.mp4 -o output.mp4
```
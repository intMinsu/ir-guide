# jetson-inference/utils/python/bindings/CMakeLists.txt
# Silence CMP0148 warning (we intentionally use legacy FindPython* modules)
# Silence CMP0148 warning (we intentionally use legacy FindPython* modules)
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

# Respect a user-provided interpreter; only clear caches if not explicitly set.
if(NOT DEFINED PYTHON_EXECUTABLE)
  unset(PYTHONINTERP_FOUND CACHE)
  unset(PYTHON_EXECUTABLE CACHE)
  unset(PYTHON_VERSION_STRING CACHE)
  unset(PYTHON_VERSION_MAJOR CACHE)
  unset(PYTHON_VERSION_MINOR CACHE)
  unset(PYTHON_VERSION_PATCH CACHE)

  unset(PYTHON_INCLUDE_PATH CACHE)
  unset(PYTHON_INCLUDE_DIRS CACHE)
  unset(PYTHON_INCLUDE_DIR CACHE)
  unset(PYTHON_LIBRARY CACHE)
  unset(PYTHON_LIBRARIES CACHE)
  unset(PYTHON_DEBUG_LIBRARIES CACHE)
  unset(PYTHON_MODULE_PREFIX CACHE)
  unset(PYTHON_MODULE_EXTENSION CACHE)

  unset(PYTHONLIBS_FOUND CACHE)
  unset(PYTHONLIBS_VERSION_STRING CACHE)
endif()

# locate requested python version
message("-- detecting Python ${PYTHON_BINDING_VERSION}...")

find_package(PythonInterp ${PYTHON_BINDING_VERSION} QUIET)
find_package(PythonLibs  ${PYTHON_BINDING_VERSION} QUIET)

if(NOT ${PYTHONLIBS_FOUND})
  message("-- Python ${PYTHON_BINDING_VERSION} wasn't found")
  return()
endif()

message("-- found Python version:  ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} (${PYTHONLIBS_VERSION_STRING})")
message("-- found Python include:  ${PYTHON_INCLUDE_DIRS}")
message("-- found Python library:  ${PYTHON_LIBRARIES}")

include_directories(${PYTHON_INCLUDE_DIRS})

# attempt to find numpy (module lives here)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
message("-- CMake module path:  ${CMAKE_MODULE_PATH}")
find_package(NumPy QUIET)

if(${NUMPY_FOUND})
  message("-- found NumPy version:  ${NUMPY_VERSION}")
  message("-- found NumPy include:  ${NUMPY_INCLUDE_DIR}")
  add_definitions(-DHAS_NUMPY)
  include_directories(${NUMPY_INCLUDE_DIR})
  link_directories(${NUMPY_INCLUDE_DIR}/../lib)
else()
  message("-- NumPy not found")
endif()

# build the bindings
file(GLOB pythonUtilsSources *.cpp)

cuda_add_library(jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
                 SHARED ${pythonUtilsSources})

# Build a single list of link libs and make ONE call (plain signature)
set(UTILS_PY_LINK_LIBS
    jetson-utils
    ${PYTHON_LIBRARIES}
)

if(${NUMPY_FOUND})
  # Only link npymath if the static archive exists
  set(NUMPY_MATH_LIB "${NUMPY_INCLUDE_DIR}/../lib/libnpymath.a")
  if(EXISTS "${NUMPY_MATH_LIB}")
    list(APPEND UTILS_PY_LINK_LIBS "${NUMPY_MATH_LIB}")
  else()
    message(STATUS "NumPy found, but libnpymath not present - skipping npymath link (OK)")
  endif()
endif()

target_link_libraries(jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
  ${UTILS_PY_LINK_LIBS}
)

set_target_properties(jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
  PROPERTIES
    PREFIX ""
    OUTPUT_NAME "jetson_utils_python"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
)

# Determine where to install Python modules:
# 1) honor -DPYTHON_INSTALL_DIR if provided
# 2) otherwise, query the chosen interpreter's site-packages (platlib)
if(DEFINED PYTHON_INSTALL_DIR AND NOT PYTHON_INSTALL_DIR STREQUAL "")
  set(PYTHON_BINDING_INSTALL_DIR "${PYTHON_INSTALL_DIR}")
else()
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['platlib'])"
    OUTPUT_VARIABLE PYTHON_BINDING_INSTALL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()
set(PYTHON_BINDING_INSTALL_DIR "${PYTHON_BINDING_INSTALL_DIR}" CACHE PATH "Where to install Python modules")
message(STATUS "Python modules will install to: ${PYTHON_BINDING_INSTALL_DIR}")

install(TARGETS jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
        DESTINATION ${PYTHON_BINDING_INSTALL_DIR})

install(DIRECTORY ../python/Jetson       DESTINATION ${PYTHON_BINDING_INSTALL_DIR})
install(DIRECTORY ../python/jetson       DESTINATION ${PYTHON_BINDING_INSTALL_DIR})
install(DIRECTORY ../python/jetson_utils DESTINATION ${PYTHON_BINDING_INSTALL_DIR})

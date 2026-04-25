find_package(OpenMP REQUIRED)

if(NOT DEFINED UPDATE_VERSION)
    set(UPDATE_VERSION "3")
endif()

if(UPDATE_VERSION STREQUAL "1")
    set(KMEANS_SRC kmeans_v1.c)
elseif(UPDATE_VERSION STREQUAL "2")
    set(KMEANS_SRC kmeans_v2.c)
elseif(UPDATE_VERSION STREQUAL "3")
    set(KMEANS_SRC kmeans_v3.c)
else()
    message(FATAL_ERROR "Unknown UPDATE_VERSION: ${UPDATE_VERSION} (expected 1|2|3)")
endif()

message(STATUS "Building CPU version: ${UPDATE_VERSION} (${KMEANS_SRC})")

add_library(optima_kmeans optima_kmeans.c dataloader.c ${KMEANS_SRC})
target_link_libraries(optima_kmeans PUBLIC OpenMP::OpenMP_C)
target_include_directories(optima_kmeans PUBLIC ${CMAKE_SOURCE_DIR}/include)

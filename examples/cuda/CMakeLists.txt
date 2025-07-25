project(Ponca_Examples_cuda LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)

    if( CMAKE_CUDA_COMPILER )

    set (CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    enable_language(CUDA)

    if(POLICY CMP0104)
        cmake_policy(SET CMP0104 NEW)
    endif()

    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()


    find_package(PNG QUIET)
    if ( PNG_FOUND )

        add_executable(ponca_ssgls "ponca_ssgls.cu")
        target_include_directories(ponca_ssgls PRIVATE ${PONCA_src_ROOT})
        target_link_libraries(ponca_ssgls PRIVATE PNG::PNG)
        add_dependencies(ponca-examples ponca_ssgls)
        set_property(TARGET ponca_ssgls PROPERTY CUDA_ARCHITECTURES OFF)
        ponca_handle_eigen_dependency(ponca_ssgls)

        # Copy assets
        add_custom_command(TARGET ponca_ssgls POST_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy_directory
                               ${CMAKE_CURRENT_SOURCE_DIR}/data
                               $<TARGET_FILE_DIR:ponca_ssgls>/data
                           COMMENT "Copying ssgls data to build tree"
                           VERBATIM
                           )
    else()
        message("LibPNG not found, skipping Ponca_ssgls")
    endif ( PNG_FOUND )

else()
    message(INFO "Cuda not found, skipping Ponca_Examples_cuda")
endif( CMAKE_CUDA_COMPILER )

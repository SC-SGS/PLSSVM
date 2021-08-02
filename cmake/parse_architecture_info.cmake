function(parse_architecture_info target_platform target_archs num_archs)
  # transform platforms to list (e.g "cuda:sm_70,sm_80" -> "cuda;sm_70,sm_80")
  string(REPLACE ":" ";" ARCH_LIST ${target_platform})
  # remove platform from list (e.g. "cuda;sm_70,sm_80" -> "sm_70,sm_80")
  list(POP_FRONT ARCH_LIST)

  if(ARCH_LIST STREQUAL "")
    # immediately return if architecture list is empty
    set(${target_archs} "" PARENT_SCOPE)
    set(${num_archs} 0 PARENT_SCOPE)
  else()
    # transform architectures to list and set output-variable (e.g. "sm_70,sm_80" -> "sm_70;sm_80")
    string(REPLACE "," ";" ARCH_LIST ${ARCH_LIST})
    set(${target_archs} ${ARCH_LIST} PARENT_SCOPE)

    # get number of architectures and set output-variable (e.g. "sm_70;sm_80" -> 2)
    list(LENGTH ARCH_LIST LEN)
    set(${num_archs} ${LEN} PARENT_SCOPE)
  endif()
endfunction()


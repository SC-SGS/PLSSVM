# set variable in the local and parent scope
macro(set_local_and_parent NAME VALUE)
    set(${ARGV0} ${ARGV1})
    set(${ARGV0} ${ARGV1} PARENT_SCOPE)
endmacro()
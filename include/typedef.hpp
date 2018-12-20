#pragma once
typedef double real_t;                              //TODO: float Fehler finden

static const unsigned CUDABLOCK_SIZE = 6;          // minimal 1
static const unsigned THREADBLOCK_SIZE = CUDABLOCK_SIZE;
// static const unsigned BLOCKING_SIZE_THREAD = ;    
static const unsigned INTERNALBLOCK_SIZE = 8;       // minimal 1


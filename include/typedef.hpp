#pragma once
typedef float real_t;                              //TODO: float Fehler finden

static const unsigned CUDABLOCK_SIZE = 6;          // minimal 1
static const unsigned THREADBLOCK_SIZE = CUDABLOCK_SIZE; //TODO: immer nur 1 mal
static const unsigned INTERNALBLOCK_SIZE = 8;       // minimal 1
static const unsigned BLOCKING_SIZE_THREAD = INTERNALBLOCK_SIZE;    


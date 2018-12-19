#pragma once
typedef double real_t;                              //TODO: float Fehler finden

static const unsigned CUDABLOCK_SIZE = 16;          // minimal 1
static const unsigned BLOCKING_SIZE_THREAD = 6;     // minimal 1 maximal CUDABLOCK_SIZE/2 //TODO: beschränkung für CUDABLOCK_SIZE/2 entfernen



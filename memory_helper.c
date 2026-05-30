/* memory_helper.c – Cache-aligned allocation helpers */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* Allocate memory on a 64-byte (cache-line) boundary */
void *aligned_alloc_64(size_t size)
{
    /* Round size up to a multiple of 64 */
    size_t padded = (size + 63) & ~(size_t)63;
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, padded) != 0)
        return NULL;
    return ptr;
}

/* Print alignment status for diagnostics */
void check_alignment(void *ptr, const char *name)
{
    uintptr_t addr = (uintptr_t)ptr;
    if (addr % 64 == 0)
        printf("[OK]    %-20s  0x%lx  — 64-byte aligned\n", name, (unsigned long)addr);
    else if (addr % 8 == 0)
        printf("[WARN]  %-20s  0x%lx  — 8-byte aligned only\n", name, (unsigned long)addr);
    else
        printf("[ERROR] %-20s  0x%lx  — NOT aligned!\n", name, (unsigned long)addr);
}

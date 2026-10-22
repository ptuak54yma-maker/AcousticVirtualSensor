ENTRY(Reset_Handler)

_Min_Heap_Size = 0;
_Min_Stack_Size = 0x2000;

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
    RAM   (xrw) : ORIGIN = 0x20000000, LENGTH = 128K
    CCM   (xrw) : ORIGIN = 0x10000000, LENGTH = 64K
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
    .isr_vector :
    {
        . = ALIGN(4);
        KEEP(*(.isr_vector))
        . = ALIGN(4);
    } > FLASH

    .text :
    {
        . = ALIGN(4);
        *(.text*)
        *(.rodata*)
        KEEP(*(.init))
        KEEP(*(.fini))
        . = ALIGN(4);
        _etext = .;
    } > FLASH

    .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > FLASH
    .ARM.exidx :
    {
        __exidx_start = .;
        *(.ARM.exidx*)
        __exidx_end = .;
    } > FLASH

    _sidata = LOADADDR(.data);
    .data :
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH

    .bss (NOLOAD) :
    {
        . = ALIGN(8);
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(8);
        _ebss = .;
    } > RAM

    .ccm (NOLOAD) :
    {
        . = ALIGN(8);
        *(.ccm*)
        . = ALIGN(8);
    } > CCM

    .heap_stack_reservation (NOLOAD) :
    {
        . = ALIGN(8);
        . += _Min_Heap_Size;
        . += _Min_Stack_Size;
        . = ALIGN(8);
    } > RAM

    /DISCARD/ : { *(.note*) *(.comment*) }
}

ASSERT(_ebss + _Min_Heap_Size + _Min_Stack_Size <= ORIGIN(RAM) + LENGTH(RAM),
       "SRAM overflow: static data and reserved stack exceed 128 KiB")
ASSERT(LOADADDR(.data) + SIZEOF(.data) <= ORIGIN(FLASH) + LENGTH(FLASH),
       "Flash overflow: firmware exceeds 1 MiB STM32F407VG flash")

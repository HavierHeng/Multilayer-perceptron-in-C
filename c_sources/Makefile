
CC=riscv64-unknown-elf-gcc
LD=riscv64-unknown-elf-ld
SREC_CAT=srec_cat
OBJCOPY=riscv64-unknown-elf-objcopy
CFLAGS=-mabi=ilp32 -march=rv32im
LDFLAGS=-Tlink.ld -melf32lriscv
SRCS=$(wildcard *.c)
OBJS=$(SRCS:.c=.o)
ELFS=$(SRCS:.c=.elf)
BINS=$(SRCS:.c=.text.vh) $(SRCS:.c=.data.vh)
VHS=$(SRCS:.c=.text.vh) $(SRCS:.c=.data.vh)

all: $(VHS)

%.o: %.c
	${CC} -c ${CFLAGS} -o $@ $<

%.elf: %.o 
	${LD} ${LDFLAGS} $< -o $@

%.text.bin: %.elf
	${OBJCOPY} -O binary $< $@ -j .init -j .text --change-addresses -0x000000

%.data.bin: %.elf
	${OBJCOPY} -O binary $< $@ -j .data -j .eh_frame -j .fini_array -j .init_array --change-addresses -0x00002000

%.vh: %.bin
	${SREC_CAT} $< -binary -byte-swap 4 -O $@ -VMem 32

clean:
	rm -f *.o
	rm -f *.elf
	rm -f *.data.bin
	rm -f *.text.bin
	rm -f *.data.vh
	rm -f *.text.vh


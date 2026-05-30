
#  Milestone-3 Makefile
#  Targets: lkf_filter   (lkf_driver.c + lkf_asm.s)
#           ekf_filter   (ekf_driver.c + ekf_asm.s)
#
#  Cross-compiler: riscv64-linux-gnu-gcc (LP64D ABI)
#  Emulator:       qemu-riscv64 (if running on non-RISC-V host)

CC      = riscv64-linux-gnu-gcc
AS      = riscv64-linux-gnu-gcc  
LD      = riscv64-linux-gnu-gcc


MARCH   = rv64imafd
MABI    = lp64d

CFLAGS  = -march=$(MARCH) -mabi=$(MABI) -O2 -Wall -Wextra \
           -ffreestanding -static -lm
ASFLAGS = -march=$(MARCH) -mabi=$(MABI) -c


QEMU    = qemu-riscv64 -L /usr/riscv64-linux-gnu


LKF_CSRC  = lkf_driver.c
LKF_SSRC  = lkf_asm.s
LKF_OBJ   = lkf_driver.o lkf_asm.o
LKF_BIN   = lkf_filter

EKF_CSRC  = ekf_driver.c
EKF_SSRC  = ekf_asm.s
EKF_OBJ   = ekf_driver.o ekf_asm.o
EKF_BIN   = ekf_filter


LKF_OUT   = LKF_Output
EKF_OUT   = EKF_Output


.PHONY: all
all: $(LKF_BIN) $(EKF_BIN)
	@echo ""
	@echo "Build complete."
	@echo "  LKF binary : $(LKF_BIN)"
	@echo "  EKF binary : $(EKF_BIN)"
	@echo "  Run with   : make run"

$(LKF_BIN): $(LKF_OBJ)
	$(LD) $(CFLAGS) -o $@ $^ -lm
	@echo "Linked: $@"

lkf_driver.o: $(LKF_CSRC)
	$(CC) $(CFLAGS) -c -o $@ $<

lkf_asm.o: $(LKF_SSRC)
	$(AS) $(ASFLAGS) -o $@ $<


$(EKF_BIN): $(EKF_OBJ)
	$(LD) $(CFLAGS) -o $@ $^ -lm
	@echo "Linked: $@"

ekf_driver.o: $(EKF_CSRC)
	$(CC) $(CFLAGS) -c -o $@ $<

ekf_asm.o: $(EKF_SSRC)
	$(AS) $(ASFLAGS) -o $@ $<

.PHONY: run run_lkf run_ekf

run: run_lkf run_ekf

run_lkf: $(LKF_BIN)
	@mkdir -p $(LKF_OUT)
	@echo "---------- Running LKF ----------"
	$(QEMU) ./$(LKF_BIN)

run_ekf: $(EKF_BIN)
	@mkdir -p $(EKF_OUT)
	@echo "---------- Running EKF ----------"
	$(QEMU) ./$(EKF_BIN)


.PHONY: native

native:
	$(MAKE) CC=gcc AS=gcc LD=gcc \
	        CFLAGS="-march=$(MARCH) -mabi=$(MABI) -O2 -Wall -lm" \
	        QEMU=""
	@echo "Native build complete."

native_run: native
	@mkdir -p $(LKF_OUT) $(EKF_OUT)
	./$(LKF_BIN)
	./$(EKF_BIN)


.PHONY: verify
verify:
	@echo "Running numerical verification..."
	python3 verify.py


.PHONY: plots
plots:
	@echo "Generating Milestone-3 plots..."
	python3 plot_m3.py
	@echo "Generating Milestone-2 vs 3 comparison plots..."
	python3 plot_compare.py


.PHONY: listing
listing:
	$(AS) $(ASFLAGS) -g -S -o lkf_asm_annotated.s $(LKF_SSRC)
	$(AS) $(ASFLAGS) -g -S -o ekf_asm_annotated.s $(EKF_SSRC)
	riscv64-linux-gnu-objdump -d lkf_asm.o > lkf_asm.lst
	riscv64-linux-gnu-objdump -d ekf_asm.o > ekf_asm.lst
	@echo "Listings: lkf_asm.lst  ekf_asm.lst"


.PHONY: clean distclean
clean:
	rm -f $(LKF_OBJ) $(EKF_OBJ)
	rm -f *.lst *.o *_annotated.s

distclean: clean
	rm -f $(LKF_BIN) $(EKF_BIN)
	rm -rf $(LKF_OUT) $(EKF_OUT)


.PHONY: help
help:
	@echo "Milestone-3 Makefile"
	@echo ""
	@echo "  make          Build LKF and EKF binaries (cross-compile)"
	@echo "  make run      Run both filters under qemu-riscv64"
	@echo "  make run_lkf  Run LKF only"
	@echo "  make run_ekf  Run EKF only"
	@echo "  make native   Build natively on a RISC-V machine"
	@echo "  make verify   Compare Milestone-2 vs Milestone-3 outputs"
	@echo "  make plots    Generate all plots"
	@echo "  make listing  Disassemble object files"
	@echo "  make clean    Remove object files"
	@echo "  make distclean Remove all generated files"

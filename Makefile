# ============================================================
# Makefile – Milestone 4: LKF + EKF on RISC-V RV64GCV
# ============================================================
CC      = riscv64-linux-gnu-gcc
CFLAGS  = -march=rv64gcv -mabi=lp64d -O2 -static -Wall
LDFLAGS = -lm

# ---- Main simulation binary ----
TARGET  = kalman_m4
OBJS    = main_driver.o lkf_vector.o ekf_vector.o memory_helper.o

# ---- Vectorised matvec unit test ----
TEST_TARGET = test_matvec
TEST_OBJS   = test_vector.o lkf_vector.o

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TEST_TARGET): $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.s
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET) $(TEST_TARGET) lkf_output.csv ekf_output.csv

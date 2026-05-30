

.section .text
.align 2


# vec_axpy_asm(double *y, double alpha, const double *x, int n)
#   y[i] += alpha * x[i],  i = 0 .. n-1
#   alpha -> fa0,  x -> a2,  n -> a3

    .globl vec_axpy_asm
    .type  vec_axpy_asm, @function
vec_axpy_asm:
    beqz    a3, vec_axpy_done
    mv      t0, a3
vec_axpy_loop:
    fld     ft0,  0(a2)
    fld     ft1,  0(a0)
    addi    a2,   a2, 8
    fmadd.d ft1,  fa0, ft0, ft1
    addi    t0,   t0, -1
    fsd     ft1,  0(a0)
    addi    a0,   a0, 8
    bnez    t0,   vec_axpy_loop
vec_axpy_done:
    ret


# mat_dot_asm(const double *A, const double *B, int n) -> double
#   accumulates A[i]*B[i] via fmadd; result in fa0

    .globl mat_dot_asm
    .type  mat_dot_asm, @function
mat_dot_asm:
    fmv.d.x fa0, zero
    beqz    a2, mat_dot_done
    mv      t0, a2
mat_dot_loop:
    fld     ft0,  0(a0)
    fld     ft1,  0(a1)
    addi    a0,   a0, 8
    addi    a1,   a1, 8
    fmadd.d fa0,  ft0, ft1, fa0
    addi    t0,   t0, -1
    bnez    t0,   mat_dot_loop
mat_dot_done:
    ret


# mat_add_asm(double *C, const double *A, const double *B, int n)
#   C[i] = A[i] + B[i]

    .globl mat_add_asm
    .type  mat_add_asm, @function
mat_add_asm:
    beqz    a3, mat_add_done
    mv      t0, a3
mat_add_loop:
    fld     ft0,  0(a1)
    fld     ft1,  0(a2)
    addi    a1,   a1, 8
    addi    a2,   a2, 8
    fadd.d  ft2,  ft0, ft1
    addi    t0,   t0, -1
    fsd     ft2,  0(a0)
    addi    a0,   a0, 8
    bnez    t0,   mat_add_loop
mat_add_done:
    ret


# mat_sub_asm(double *C, const double *A, const double *B, int n)
#   C[i] = A[i] - B[i]

    .globl mat_sub_asm
    .type  mat_sub_asm, @function
mat_sub_asm:
    beqz    a3, mat_sub_done
    mv      t0, a3
mat_sub_loop:
    fld     ft0,  0(a1)
    fld     ft1,  0(a2)
    addi    a1,   a1, 8
    addi    a2,   a2, 8
    fsub.d  ft2,  ft0, ft1
    addi    t0,   t0, -1
    fsd     ft2,  0(a0)
    addi    a0,   a0, 8
    bnez    t0,   mat_sub_loop
mat_sub_done:
    ret


# mat_scale_asm(double *A, double scalar, int n)
#   in-place scale: A[i] *= scalar
#   scalar -> fa0,  n -> a1

    .globl mat_scale_asm
    .type  mat_scale_asm, @function
mat_scale_asm:
    beqz    a1, mat_scale_done
    mv      t0, a1
mat_scale_loop:
    fld     ft0,  0(a0)
    addi    t0,   t0, -1
    fmul.d  ft0,  ft0, fa0
    fsd     ft0,  0(a0)
    addi    a0,   a0, 8
    bnez    t0,   mat_scale_loop
mat_scale_done:
    ret


# predict_x_asm(double *x12, double dt)
#
#   Propagates a 12-element joint state vector through the
#   constant-jerk transition matrix F (block-diagonal, 3 axes).
#   For each axis block b (0..2), base offset o = b*4:
#
#     pos  += dt*vel + (dt^2/2)*acc + (dt^3/6)*jerk
#     vel  +=          dt*acc       + (dt^2/2)*jerk
#     acc  +=                         dt*jerk
#     jerk  = unchanged
#
#   a0  = double *x12
#   fa0 = dt
#
#   Callee-saved: fs0=dt  fs1=dt^2/2  fs2=dt^3/6

    .globl predict_x_asm
    .type  predict_x_asm, @function
predict_x_asm:
    addi    sp,  sp, -32
    sd      ra,  24(sp)
    fsd     fs0, 16(sp)
    fsd     fs1,  8(sp)
    fsd     fs2,  0(sp)

    fmv.d   fs0, fa0              # fs0 = dt

    la      t0,  const_half
    fld     ft6,  0(t0)
    fmul.d  fs1,  fs0, fs0        # dt^2
    fmul.d  fs1,  fs1, ft6        # fs1 = dt^2/2

    la      t0,  const_third
    fld     ft6,  0(t0)
    fmul.d  fs2,  fs1, fs0        # dt^3/2
    fmul.d  fs2,  fs2, ft6        # fs2 = dt^3/6

    li      t6, 0
pxa_axis:
    li      t5, 3
    bge     t6, t5, pxa_done

    slli    t0, t6, 5
    add     t0, a0, t0            # t0 = &x[b*4]

    fld     ft0,  0(t0)           # pos  (original)
    fld     ft1,  8(t0)           # vel  (original)
    fld     ft2, 16(t0)           # acc  (original)
    fld     ft3, 24(t0)           # jerk (read-only)

    # new acc = acc + dt*jerk
    fmadd.d ft4, fs0, ft3, ft2

    # new vel = vel + dt*acc + dt2h*jerk
    fmadd.d ft5, fs0, ft2, ft1
    fmadd.d ft5, fs1, ft3, ft5

    # new pos = pos + dt*vel + dt2h*acc + dt3s*jerk
    fmadd.d ft6, fs0, ft1, ft0
    fmadd.d ft6, fs1, ft2, ft6
    fmadd.d ft6, fs2, ft3, ft6

    fsd     ft6,  0(t0)
    fsd     ft5,  8(t0)
    fsd     ft4, 16(t0)

    addi    t6, t6, 1
    j       pxa_axis
pxa_done:
    fld     fs2,  0(sp)
    fld     fs1,  8(sp)
    fld     fs0, 16(sp)
    ld      ra,  24(sp)
    addi    sp,  sp, 32
    ret


# update_x_asm(double *x12, const double *K_row3, const double *innov3)
#
#   Applies the Kalman correction step to one joint:
#     x[i] += K[i][0]*innov[0] + K[i][1]*innov[1] + K[i][2]*innov[2]
#   for i = 0 .. 11
#
#   K_row3 is a 12x3 submatrix stored row-major (stride = 24 bytes).
#   All three innovation values are pre-loaded before the loop.

    .globl update_x_asm
    .type  update_x_asm, @function
update_x_asm:
    addi    sp,  sp, -16
    sd      s0,   0(sp)
    sd      ra,   8(sp)

    fld     ft3,  0(a2)           # innov[0]
    fld     ft4,  8(a2)           # innov[1]
    fld     ft5, 16(a2)           # innov[2]

    mv      s0, a0
    li      t6, 0
uxa_loop:
    li      t5, 12
    bge     t6, t5, uxa_done

    li      t0, 24
    mul     t0, t6, t0
    add     t0, a1, t0

    fld     ft0,  0(t0)
    fld     ft1,  8(t0)
    fld     ft2, 16(t0)

    fmul.d  ft6, ft0, ft3
    fmadd.d ft6, ft1, ft4, ft6
    fmadd.d ft6, ft2, ft5, ft6

    slli    t1, t6, 3
    add     t1, s0, t1
    fld     ft0,  0(t1)
    fadd.d  ft0,  ft0, ft6
    addi    t6,   t6, 1
    fsd     ft0,  0(t1)
    j       uxa_loop
uxa_done:
    ld      s0,   0(sp)
    ld      ra,   8(sp)
    addi    sp,  sp, 16
    ret


# covar_diag_update_asm(double *P12diag, double factor, int n)
#   Scales n diagonal elements by factor: P[i] *= factor
#   factor -> fa0,  n -> a1

    .globl covar_diag_update_asm
    .type  covar_diag_update_asm, @function
covar_diag_update_asm:
    beqz    a1, cdu_done
    mv      t0, a1
cdu_loop:
    fld     ft0,  0(a0)
    addi    t0,   t0, -1
    fmul.d  ft0,  ft0, fa0
    fsd     ft0,  0(a0)
    addi    a0,   a0, 8
    bnez    t0,   cdu_loop
cdu_done:
    ret


# Constants (read-only, 8-byte aligned)

    .section .rodata
    .align 3
const_half:
    .double 0.5
const_third:
    .double 0.3333333333333333
const_one:
    .double 1.0
const_1em12:
    .double 1.0e-12
const_1em14:
    .double 1.0e-14
const_neg1em14:
    .double -1.0e-14

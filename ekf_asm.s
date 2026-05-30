


    .section .text
    .align 2


# predict_x_asm(double *x12, double dt)
#
#   Propagates one joint's 12-element state vector through the
#   constant-jerk transition. For each axis block b (0..2),
#   base offset o = b*4:
#     x[o]   += dt*x[o+1] + dt²/2*x[o+2] + dt³/6*x[o+3]
#     x[o+1] +=             dt  *x[o+2] + dt²/2*x[o+3]
#     x[o+2] +=                            dt  *x[o+3]
#
#   a0 = double *x12,  fa0 = dt
#   Callee-saved: fs0=dt  fs1=dt²/2  fs2=dt³/6

    .globl predict_x_asm
    .type  predict_x_asm, @function
predict_x_asm:
    addi    sp, sp, -32
    sd      ra, 24(sp)
    fsd     fs0, 16(sp)
    fsd     fs1,  8(sp)
    fsd     fs2,  0(sp)

    fmv.d   fs0, fa0              # dt

    fmul.d  fs1, fs0, fs0
    la      t0, const_half
    fld     ft6, 0(t0)
    fmul.d  fs1, fs1, ft6         # dt²/2

    fmul.d  fs2, fs1, fs0
    la      t0, const_third
    fld     ft6, 0(t0)
    fmul.d  fs2, fs2, ft6         # dt³/6

    li      t6, 0
pxa_axis:
    li      t5, 3
    bge     t6, t5, pxa_done

    slli    t0, t6, 5
    add     t0, a0, t0

    fld     ft0,  0(t0)           # pos
    fld     ft1,  8(t0)           # vel
    fld     ft2, 16(t0)           # acc
    fld     ft3, 24(t0)           # jerk

    fmadd.d ft0, fs0, ft1, ft0
    fmadd.d ft0, fs1, ft2, ft0
    fmadd.d ft0, fs2, ft3, ft0
    fsd     ft0,  0(t0)

    fmadd.d ft1, fs0, ft2, ft1
    fmadd.d ft1, fs1, ft3, ft1
    fsd     ft1,  8(t0)

    fmadd.d ft2, fs0, ft3, ft2
    fsd     ft2, 16(t0)

    addi    t6, t6, 1
    j       pxa_axis
pxa_done:
    fld     fs0, 16(sp)
    fld     fs1,  8(sp)
    fld     fs2,  0(sp)
    ld      ra,  24(sp)
    addi    sp, sp, 32
    ret

# ekf_update_x_asm(double *x12, const double *K12x3,
#                  const double *innov3)
#
#   Applies Kalman correction to one joint's 12-state vector:
#     x12[i] += K[i][0]*innov[0] + K[i][1]*innov[1] + K[i][2]*innov[2]
#   K is stored row-major (12 rows x 3 cols, stride = 24 bytes).
#
#   a0 = x12,  a1 = K (12x3 row-major),  a2 = innov3

    .globl ekf_update_x_asm
    .type  ekf_update_x_asm, @function
ekf_update_x_asm:
    addi    sp, sp, -16
    sd      ra,  8(sp)
    sd      s0,  0(sp)

    mv      s0, a0
    li      t6, 0
ekfu_loop:
    li      t5, 12
    bge     t6, t5, ekfu_done

    li      t0, 24
    mul     t0, t6, t0
    add     t0, a1, t0

    fld     ft0,  0(t0)
    fld     ft3,  0(a2)
    fmul.d  ft6, ft0, ft3

    fld     ft1,  8(t0)
    fld     ft4,  8(a2)
    fmadd.d ft6, ft1, ft4, ft6

    fld     ft2, 16(t0)
    fld     ft5, 16(a2)
    fmadd.d ft6, ft2, ft5, ft6

    slli    t1, t6, 3
    add     t1, s0, t1
    fld     ft0, 0(t1)
    fadd.d  ft0, ft0, ft6
    fsd     ft0, 0(t1)

    addi    t6, t6, 1
    j       ekfu_loop
ekfu_done:
    ld      ra,  8(sp)
    ld      s0,  0(sp)
    addi    sp, sp, 16
    ret

# atan_poly_asm(double x) -> fa0
#
#   Evaluates the A&S 4.4.49 minimax atan polynomial via Horner:
#     p(x) = x*(c0 + x²*(c1 + x²*(... + x²*c8)))
#   All multiply-accumulate steps use fmadd.d.
#
#   fa0 = x (input and output)
#   ft0 = x²,  ft1 = Horner accumulator,  ft2-ft10 = c0..c8

    .globl atan_poly_asm
    .type  atan_poly_asm, @function
atan_poly_asm:
    fmul.d  ft0, fa0, fa0         # x²

    la      t0, atan_c
    fld     ft2,  0(t0)           # c0
    fld     ft3,  8(t0)           # c1
    fld     ft4, 16(t0)           # c2
    fld     ft5, 24(t0)           # c3
    fld     ft6, 32(t0)           # c4
    fld     ft7, 40(t0)           # c5
    fld     ft8, 48(t0)           # c6
    fld     ft9, 56(t0)           # c7
    fld     ft10,64(t0)           # c8

    fmv.d   ft1, ft10             # accumulator = c8
    fmadd.d ft1, ft1, ft0, ft9
    fmadd.d ft1, ft1, ft0, ft8
    fmadd.d ft1, ft1, ft0, ft7
    fmadd.d ft1, ft1, ft0, ft6
    fmadd.d ft1, ft1, ft0, ft5
    fmadd.d ft1, ft1, ft0, ft4
    fmadd.d ft1, ft1, ft0, ft3
    fmadd.d ft1, ft1, ft0, ft2

    fmul.d  fa0, ft1, fa0         # p(x)*x
    ret


# atan2_asm(double y, double x) -> fa0
#
#   Full four-quadrant atan2 built on atan_poly_asm.
#   fa0 = y,  fa1 = x
#
#   Algorithm:
#     if x == 0: return sign-conditional pi/2 or 0
#     ax = |x|,  ay = |y|
#     if ax >= ay: a = atan_poly(ay / (ax + eps))
#     else:        a = pi/2 - atan_poly(ax / (ay + eps))
#     if x < 0:   a = pi - a
#     if y < 0:   a = -a
#
#   fs0=y  fs1=x  fs2=|x|  fs3=|y|  fs4=result

    .globl atan2_asm
    .type  atan2_asm, @function
atan2_asm:
    addi    sp, sp, -48
    sd      ra, 40(sp)
    fsd     fs0, 32(sp)
    fsd     fs1, 24(sp)
    fsd     fs2, 16(sp)
    fsd     fs3,  8(sp)
    fsd     fs4,  0(sp)

    fmv.d   fs0, fa0
    fmv.d   fs1, fa1

    fmv.d.x ft0, zero
    feq.d   t0, fs1, ft0
    beqz    t0, at2_nonzero

    la      t1, const_pi_half
    fld     ft1, 0(t1)
    flt.d   t0, ft0, fs0
    beqz    t0, at2_x0_neg
    fmv.d   fa0, ft1
    j       at2_done
at2_x0_neg:
    flt.d   t0, fs0, ft0
    beqz    t0, at2_x0_zero
    fneg.d  fa0, ft1
    j       at2_done
at2_x0_zero:
    fmv.d.x fa0, zero
    j       at2_done

at2_nonzero:
    fsgnjx.d fs2, fs1, fs1
    fsgnjx.d fs3, fs0, fs0

    la      t0, const_eps300
    fld     ft2, 0(t0)

    fle.d   t0, fs3, fs2
    beqz    t0, at2_else

    fadd.d  ft0, fs2, ft2
    fdiv.d  fa0, fs3, ft0
    call    atan_poly_asm
    fmv.d   fs4, fa0
    j       at2_quad

at2_else:
    la      t0, const_pi_half
    fld     ft1, 0(t0)
    fadd.d  ft0, fs3, ft2
    fdiv.d  fa0, fs2, ft0
    call    atan_poly_asm
    fsub.d  fs4, ft1, fa0

at2_quad:
    fmv.d.x ft0, zero
    la      t0, const_pi
    fld     ft1, 0(t0)

    flt.d   t0, fs1, ft0
    beqz    t0, at2_check_y
    fsub.d  fs4, ft1, fs4

at2_check_y:
    fmv.d.x ft0, zero
    flt.d   t0, fs0, ft0
    beqz    t0, at2_result
    fneg.d  fs4, fs4

at2_result:
    fmv.d   fa0, fs4

at2_done:
    ld      ra, 40(sp)
    fld     fs0, 32(sp)
    fld     fs1, 24(sp)
    fld     fs2, 16(sp)
    fld     fs3,  8(sp)
    fld     fs4,  0(sp)
    addi    sp, sp, 48
    ret


# wrap_angle_asm(double v) -> fa0
#   Iteratively wraps v into the half-open interval (-pi, pi].
#   ft0 = 2*pi,  ft1 = pi,  ft2 = -pi

    .globl wrap_angle_asm
    .type  wrap_angle_asm, @function
wrap_angle_asm:
    la      t0, const_two_pi
    fld     ft0, 0(t0)
    la      t0, const_pi
    fld     ft1, 0(t0)
    fneg.d  ft2, ft1
wrap_pos:
    flt.d   t0, ft1, fa0
    beqz    t0, wrap_neg
    fsub.d  fa0, fa0, ft0
    j       wrap_pos
wrap_neg:
    flt.d   t0, fa0, ft2
    beqz    t0, wrap_done
    fadd.d  fa0, fa0, ft0
    j       wrap_neg
wrap_done:
    ret


# h_func_asm(double px, double py, double pz, double *z_sph)
#
#   Converts one joint's Cartesian position to spherical coords:
#     z_sph[0] = r     = sqrt(px²+py²+pz²) + eps
#     z_sph[1] = theta = atan2(py, px)
#     z_sph[2] = phi   = atan2(pz, rxy)
#   where rxy = sqrt(px²+py²) + eps
#
#   fa0=px  fa1=py  fa2=pz  a0=double *z_sph
#   fs0=px  fs1=py  fs2=pz  fs3=rxy  s0=z_sph ptr

    .globl h_func_asm
    .type  h_func_asm, @function
h_func_asm:
    addi    sp, sp, -48
    sd      ra, 40(sp)
    sd      s0, 32(sp)
    fsd     fs0, 24(sp)
    fsd     fs1, 16(sp)
    fsd     fs2,  8(sp)
    fsd     fs3,  0(sp)

    fmv.d   fs0, fa0
    fmv.d   fs1, fa1
    fmv.d   fs2, fa2
    mv      s0,  a0

    la      t0, const_eps300
    fld     ft3, 0(t0)

    fmul.d  ft0, fs0, fs0
    fmadd.d ft0, fs1, fs1, ft0
    fsqrt.d ft0, ft0
    fadd.d  fs3, ft0, ft3         # rxy

    fmul.d  ft1, fs0, fs0
    fmadd.d ft1, fs1, fs1, ft1
    fmadd.d ft1, fs2, fs2, ft1
    fsqrt.d ft1, ft1
    fadd.d  ft1, ft1, ft3         # r

    fsd     ft1, 0(s0)            # z_sph[0] = r

    fmv.d   fa0, fs1
    fmv.d   fa1, fs0
    call    atan2_asm
    fsd     fa0, 8(s0)            # z_sph[1] = theta

    fmv.d   fa0, fs2
    fmv.d   fa1, fs3
    call    atan2_asm
    fsd     fa0, 16(s0)           # z_sph[2] = phi

    ld      ra, 40(sp)
    ld      s0, 32(sp)
    fld     fs0, 24(sp)
    fld     fs1, 16(sp)
    fld     fs2,  8(sp)
    fld     fs3,  0(sp)
    addi    sp, sp, 48
    ret


# build_jblock_asm(double *Hj36, double px, double py, double pz)
#
#   Fills a zeroed 3x12 Jacobian block (row-major) for one joint.
#   Only the 8 position-column entries are non-zero:
#
#     J[0][0] =  px/r          J[0][4] =  py/r          J[0][8] =  pz/r
#     J[1][0] = -py/rxy²       J[1][4] =  px/rxy²
#     J[2][0] = -px*pz/(r²*rxy) J[2][4] = -py*pz/(r²*rxy) J[2][8] = rxy/r²
#
#   a0=Hj36  fa0=px  fa1=py  fa2=pz
#   s0=Hj    fs0=px  fs1=py  fs2=pz
#   fs3=rxy²  fs4=rxy  fs5=r²

    .globl build_jblock_asm
    .type  build_jblock_asm, @function
build_jblock_asm:
    addi    sp, sp, -64
    sd      ra, 56(sp)
    sd      s0, 48(sp)
    fsd     fs0, 40(sp)
    fsd     fs1, 32(sp)
    fsd     fs2, 24(sp)
    fsd     fs3, 16(sp)
    fsd     fs4,  8(sp)
    fsd     fs5,  0(sp)

    mv      s0, a0
    fmv.d   fs0, fa0
    fmv.d   fs1, fa1
    fmv.d   fs2, fa2

    la      t0, const_eps300
    fld     ft6, 0(t0)

    # Zero all 36 entries first
    fmv.d.x ft0, zero
    li      t1, 0
jblk_zero:
    li      t2, 36
    bge     t1, t2, jblk_zero_done
    slli    t2, t1, 3
    add     t2, s0, t2
    fsd     ft0, 0(t2)
    addi    t1, t1, 1
    j       jblk_zero
jblk_zero_done:

    fmul.d  ft0, fs0, fs0
    fmadd.d fs3, fs1, fs1, ft0
    fadd.d  fs3, fs3, ft6         # rxy² + eps

    fsqrt.d fs4, fs3              # rxy

    fmul.d  ft0, fs2, fs2
    fadd.d  fs5, fs3, ft0
    fadd.d  fs5, fs5, ft6         # r² + eps

    fsqrt.d ft5, fs5              # r

    # Row 0: J[0][0], J[0][4], J[0][8]  (byte offsets 0, 32, 64)
    fdiv.d  ft0, fs0, ft5
    fsd     ft0,   0(s0)
    fdiv.d  ft0, fs1, ft5
    fsd     ft0,  32(s0)
    fdiv.d  ft0, fs2, ft5
    fsd     ft0,  64(s0)

    # Row 1: J[1][0], J[1][4]  (byte offsets 96, 128)
    fdiv.d  ft0, fs1, fs3
    fneg.d  ft0, ft0
    fsd     ft0,  96(s0)
    fdiv.d  ft0, fs0, fs3
    fsd     ft0, 128(s0)

    # Row 2: J[2][0], J[2][4], J[2][8]  (byte offsets 192, 224, 256)
    fmul.d  ft1, fs5, fs4
    fmul.d  ft0, fs0, fs2
    fdiv.d  ft0, ft0, ft1
    fneg.d  ft0, ft0
    fsd     ft0, 192(s0)
    fmul.d  ft0, fs1, fs2
    fdiv.d  ft0, ft0, ft1
    fneg.d  ft0, ft0
    fsd     ft0, 224(s0)
    fdiv.d  ft0, fs4, fs5
    fsd     ft0, 256(s0)

    ld      ra, 56(sp)
    ld      s0, 48(sp)
    fld     fs0, 40(sp)
    fld     fs1, 32(sp)
    fld     fs2, 24(sp)
    fld     fs3, 16(sp)
    fld     fs4,  8(sp)
    fld     fs5,  0(sp)
    addi    sp, sp, 64
    ret


# inv3x3_asm(double *Ainv9, const double *A9, int *ok)
#
#   Exact 3x3 inversion via Cramer's rule. Sets *ok=1 on
#   success, *ok=0 if |det| < 1e-15 (caller should fall back).
#   A = [a b c; d e f; g h k] stored row-major.
#
#   a0=Ainv  a1=A  a2=*ok
#   fs0=a  fs1=b  fs2=c  fs3=d  fs4=e  fs5=f
#   ft0=g  ft1=h  ft2=k  (loaded once, kept in caller-saved)

    .globl inv3x3_asm
    .type  inv3x3_asm, @function
inv3x3_asm:
    addi    sp, sp, -80
    sd      ra,  72(sp)
    sd      s0,  64(sp)
    sd      s1,  56(sp)
    sd      s2,  48(sp)
    fsd     fs0, 40(sp)
    fsd     fs1, 32(sp)
    fsd     fs2, 24(sp)
    fsd     fs3, 16(sp)
    fsd     fs4,  8(sp)
    fsd     fs5,  0(sp)

    mv      s0, a0
    mv      s1, a1
    mv      s2, a2

    fld     fs0,  0(s1)           # a
    fld     fs1,  8(s1)           # b
    fld     fs2, 16(s1)           # c
    fld     fs3, 24(s1)           # d
    fld     fs4, 32(s1)           # e
    fld     fs5, 40(s1)           # f
    fld     ft0, 48(s1)           # g
    fld     ft1, 56(s1)           # h
    fld     ft2, 64(s1)           # k

    # det = a*(e*k - f*h) - b*(d*k - f*g) + c*(d*h - e*g)
    fmul.d  ft3, fs4, ft2
    fmul.d  ft4, fs5, ft1
    fsub.d  ft3, ft3, ft4
    fmul.d  ft3, fs0, ft3         # a*(ek-fh)

    fmul.d  ft4, fs3, ft2
    fmul.d  ft5, fs5, ft0
    fsub.d  ft4, ft4, ft5
    fmul.d  ft4, fs1, ft4         # b*(dk-fg)

    fmul.d  ft5, fs3, ft1
    fmul.d  ft6, fs4, ft0
    fsub.d  ft5, ft5, ft6
    fmul.d  ft5, fs2, ft5         # c*(dh-eg)

    fsub.d  ft3, ft3, ft4
    fadd.d  ft3, ft3, ft5         # det

    la      t0, const_1em15
    fld     ft4, 0(t0)
    fsgnjx.d ft5, ft3, ft3
    flt.d   t0, ft5, ft4
    bnez    t0, inv3_sing

    la      t0, const_one
    fld     ft6, 0(t0)
    fdiv.d  ft6, ft6, ft3         # 1/det

    # Cofactor matrix entries (row-major order)
    fmul.d  ft3, fs4, ft2; fmul.d ft4, fs5, ft1; fsub.d ft3,ft3,ft4; fmul.d ft3,ft6,ft3; fsd ft3,  0(s0)
    fmul.d  ft3, fs2, ft1; fmul.d ft4, fs1, ft2; fsub.d ft3,ft3,ft4; fmul.d ft3,ft6,ft3; fsd ft3,  8(s0)
    fmul.d  ft3, fs1, fs5; fmul.d ft4, fs2, fs4; fsub.d ft3,ft3,ft4; fmul.d ft3,ft6,ft3; fsd ft3, 16(s0)
    fmul.d  ft3, fs5, ft0; fmul.d ft4, fs3, ft2; fsub.d ft3,ft3,ft4; fmul.d ft3,ft6,ft3; fsd ft3, 24(s0)
    fmul.d  ft3, fs0, ft2; fmul.d ft4, fs2, ft0; fsub.d ft3,ft3,ft4; fmul.d ft3,ft6,ft3; fsd ft3, 32(s0)
    fmul.d  ft3, fs2, fs3; fmul.d ft4, fs0, fs5; fsub.d ft3,ft3,ft4; fmul.d ft3,ft6,ft3; fsd ft3, 40(s0)
    fmul.d  ft3, fs3, ft1; fmul.d ft4, fs4, ft0; fsub.d ft3,ft3,ft4; fmul.d ft3,ft6,ft3; fsd ft3, 48(s0)
    fmul.d  ft3, fs1, ft0; fmul.d ft4, fs0, ft1; fsub.d ft3,ft3,ft4; fmul.d ft3,ft6,ft3; fsd ft3, 56(s0)
    fmul.d  ft3, fs0, fs4; fmul.d ft4, fs1, fs3; fsub.d ft3,ft3,ft4; fmul.d ft3,ft6,ft3; fsd ft3, 64(s0)

    li      t0, 1
    sw      t0, 0(s2)
    j       inv3_done

inv3_sing:
    li      t0, 0
    sw      t0, 0(s2)

inv3_done:
    ld      ra,  72(sp)
    ld      s0,  64(sp)
    ld      s1,  56(sp)
    ld      s2,  48(sp)
    fld     fs0, 40(sp)
    fld     fs1, 32(sp)
    fld     fs2, 24(sp)
    fld     fs3, 16(sp)
    fld     fs4,  8(sp)
    fld     fs5,  0(sp)
    addi    sp, sp, 80
    ret


# mat_add_asm(double *C, const double *A, const double *B, int n)
#   C[i] = A[i] + B[i]

    .globl mat_add_asm
    .type  mat_add_asm, @function
mat_add_asm:
    beqz  a3, mat_add_done
    mv    t0, a3
mat_add_loop:
    fld   ft0, 0(a1)
    fld   ft1, 0(a2)
    fadd.d ft2, ft0, ft1
    fsd   ft2, 0(a0)
    addi  a0, a0, 8
    addi  a1, a1, 8
    addi  a2, a2, 8
    addi  t0, t0, -1
    bnez  t0, mat_add_loop
mat_add_done:
    ret


# mat_sub_asm(double *C, const double *A, const double *B, int n)
#   C[i] = A[i] - B[i]

    .globl mat_sub_asm
    .type  mat_sub_asm, @function
mat_sub_asm:
    beqz  a3, mat_sub_done
    mv    t0, a3
mat_sub_loop:
    fld   ft0, 0(a1)
    fld   ft1, 0(a2)
    fsub.d ft2, ft0, ft1
    fsd   ft2, 0(a0)
    addi  a0, a0, 8
    addi  a1, a1, 8
    addi  a2, a2, 8
    addi  t0, t0, -1
    bnez  t0, mat_sub_loop
mat_sub_done:
    ret


# mat_scale_asm(double *A, double scalar, int n)
#   A[i] *= scalar  (scalar in fa0, n in a1)

    .globl mat_scale_asm
    .type  mat_scale_asm, @function
mat_scale_asm:
    beqz  a1, mat_scale_done
    mv    t0, a1
mat_scale_loop:
    fld   ft0, 0(a0)
    fmul.d ft0, ft0, fa0
    fsd   ft0, 0(a0)
    addi  a0, a0, 8
    addi  t0, t0, -1
    bnez  t0, mat_scale_loop
mat_scale_done:
    ret


# mat_dot_asm(const double *A, const double *B, int n) -> fa0
#   Returns sum of A[i]*B[i]; uses fmadd.d for accumulation.

    .globl mat_dot_asm
    .type  mat_dot_asm, @function
mat_dot_asm:
    fmv.d.x fa0, zero
    beqz  a2, mat_dot_done
    mv    t0, a2
mat_dot_loop:
    fld   ft0, 0(a0)
    fld   ft1, 0(a1)
    fmadd.d fa0, ft0, ft1, fa0
    addi  a0, a0, 8
    addi  a1, a1, 8
    addi  t0, t0, -1
    bnez  t0, mat_dot_loop
mat_dot_done:
    ret

# vec_axpy_asm(double *y, double alpha, const double *x, int n)
#   y[i] += alpha * x[i]  (alpha in fa0, x in a2, n in a3)

    .globl vec_axpy_asm
    .type  vec_axpy_asm, @function
vec_axpy_asm:
    beqz  a3, vec_axpy_done
    mv    t0, a3
vec_axpy_loop:
    fld   ft0, 0(a2)
    fld   ft1, 0(a0)
    fmadd.d ft1, fa0, ft0, ft1
    fsd   ft1, 0(a0)
    addi  a0, a0, 8
    addi  a2, a2, 8
    addi  t0, t0, -1
    bnez  t0, vec_axpy_loop
vec_axpy_done:
    ret


# Read-only data

    .section .rodata
    .align 3

atan_c:
    .double  1.0
    .double -0.3333314528
    .double  0.1999355085
    .double -0.1420889944
    .double  0.1065626393
    .double -0.0752896400
    .double  0.0429096138
    .double -0.0161657367
    .double  0.0028662257

const_pi:       .double 3.14159265358979323846
const_pi_half:  .double 1.57079632679489661923
const_two_pi:   .double 6.28318530717958647692
const_eps300:   .double 1.0e-300
const_1em15:    .double 1.0e-15
const_one:      .double 1.0
const_half:     .double 0.5
const_third:    .double 0.3333333333333333

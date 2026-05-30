# ==============================================================
#  lkf_vector.s  –  Linear Kalman Filter (RISC-V RV64GCV)
#
#  State vector x: [px, py, vx, vy]  (N=4)
#  All matrices stored row-major, double precision (64-bit)
#
#  Globals exposed:
#    lkf_predict  – x = F*x + B*u,   P = F*P*Fᵀ + Q
#    lkf_update   – K = P*Hᵀ*(H*P*Hᵀ+R)⁻¹,  x += K*(z-H*x),  P = (I-K*H)*P
#    matvec_vec   – vectorised matrix-vector multiply (N×N · N)
#    mat_add_vec  – vectorised element-wise add
# ==============================================================

.section .data
.align 3

# ---- State vector x (4 doubles) ----
.globl lkf_x
lkf_x:   .double 0.0, 0.0, 0.0, 0.0

# ---- Process covariance P (4×4, row-major) ----
.globl lkf_P
lkf_P:
    .double 1.0, 0.0, 0.0, 0.0
    .double 0.0, 1.0, 0.0, 0.0
    .double 0.0, 0.0, 1.0, 0.0
    .double 0.0, 0.0, 0.0, 1.0

# ---- State transition F (4×4) ----
# dt = 0.1 s  →  F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
.globl lkf_F
lkf_F:
    .double 1.0, 0.0, 0.1, 0.0
    .double 0.0, 1.0, 0.0, 0.1
    .double 0.0, 0.0, 1.0, 0.0
    .double 0.0, 0.0, 0.0, 1.0

# ---- Process noise Q (4×4, small diagonal) ----
.globl lkf_Q
lkf_Q:
    .double 1e-4, 0.0,  0.0,  0.0
    .double 0.0,  1e-4, 0.0,  0.0
    .double 0.0,  0.0,  1e-2, 0.0
    .double 0.0,  0.0,  0.0,  1e-2

# ---- Measurement matrix H (2×4): observe px, py only ----
.globl lkf_H
lkf_H:
    .double 1.0, 0.0, 0.0, 0.0
    .double 0.0, 1.0, 0.0, 0.0

# ---- Measurement noise R (2×2) ----
.globl lkf_R
lkf_R:
    .double 0.1, 0.0
    .double 0.0, 0.1

# ---- Simulated measurement z (2 doubles) ----
.globl lkf_z
lkf_z:  .double 1.0, 1.0

# ---- Scratch buffers ----
.globl lkf_tmp4
lkf_tmp4:  .double 0.0, 0.0, 0.0, 0.0   # length-4 scratch

.globl lkf_tmp16
lkf_tmp16: .fill 16, 8, 0               # 4×4 scratch matrix

.globl lkf_Fx
lkf_Fx:    .double 0.0, 0.0, 0.0, 0.0   # F*x result

.globl lkf_FP
lkf_FP:   .fill 16, 8, 0                # F*P

.globl lkf_FPFt
lkf_FPFt: .fill 16, 8, 0                # F*P*Fᵀ

# --- Update temporaries (2-element) ---
.globl lkf_Hx
lkf_Hx:   .double 0.0, 0.0             # H*x  (2-vec)
.globl lkf_innov
lkf_innov:.double 0.0, 0.0             # z - H*x
.globl lkf_PHt
lkf_PHt:  .fill 8, 8, 0               # P*Hᵀ  (4×2)
.globl lkf_S
lkf_S:    .fill 4, 8, 0               # H*P*Hᵀ + R (2×2)
.globl lkf_Sinv
lkf_Sinv: .fill 4, 8, 0               # S⁻¹ (2×2)
.globl lkf_K
lkf_K:    .fill 8, 8, 0               # Kalman gain (4×2)
.globl lkf_Kv
lkf_Kv:   .double 0.0, 0.0, 0.0, 0.0  # K*innov (4-vec)
.globl lkf_KH
lkf_KH:   .fill 16, 8, 0              # K*H (4×4)
.globl lkf_IKH
lkf_IKH:  .fill 16, 8, 0              # (I - K*H) (4×4)

# ---- Frame counter / output bookkeeping ----
.globl lkf_frame
lkf_frame: .quad 0

# ---- CSV file pointer (FILE*) ----
.globl lkf_fptr
lkf_fptr: .quad 0

# ---------------------------------------------------------------
.section .text
.align 2

# ==============================================================
# Helper: matvec_NxM(out, mat, vec, rows, cols)
#   a0=out(rows), a1=mat(rows×cols), a2=vec(cols), a3=rows, a4=cols
#   Computes:  out[i] = sum_j mat[i*cols+j]*vec[j]
#   Uses RVV for the inner dot-product reduction.
# ==============================================================
.globl matvec_NxM
matvec_NxM:
    # a3 = rows remaining
    beqz a3, matvec_done
matvec_row:
    # inner dot-product: row_ptr=a1, vec=a2, len=a4
    mv   t2, a4           # elements left in this row
    mv   t3, a2           # vec pointer (reset per row)
    mv   t4, a1           # row pointer
    fmv.d.x ft0, zero     # accumulator = 0.0
dot_loop:
    vsetvli t0, t2, e64, m1, ta, ma
    fmv.d.x ft3, zero
    vfmv.s.f v0, ft3      # identity=0.0; must follow vsetvli e64
    vle64.v v1, (t4)      # load row chunk
    vle64.v v2, (t3)      # load vec chunk
    vfmul.vv v3, v1, v2   # element-wise product
    vfredusum.vs v4, v3, v0  # reduction (need v0=0 initially)
    # accumulate scalar
    vfmv.f.s ft1, v4
    fadd.d  ft0, ft0, ft1
    slli t5, t0, 3
    add  t4, t4, t5
    add  t3, t3, t5
    sub  t2, t2, t0
    bnez t2, dot_loop
    # store result
    fsd  ft0, 0(a0)
    addi a0, a0, 8        # next output element
    # advance mat row pointer by cols*8 bytes
    slli t5, a4, 3
    add  a1, a1, t5
    addi a3, a3, -1
    bnez a3, matvec_row
matvec_done:
    ret

# ==============================================================
# Helper: matTvec_NxM(out, mat, vec, rows, cols)
#   Multiply matᵀ (cols×rows transposed) by vec(rows)
#   out[j] = sum_i mat[i*cols+j]*vec[i]
#   a0=out(cols), a1=mat(rows×cols), a2=vec(rows), a3=rows, a4=cols
# ==============================================================
.globl matTvec_NxM
matTvec_NxM:
    # Zero out output first (cols elements)
    mv   t6, a4
    mv   t5, a0
zero_out:
    beqz t6, zero_done
    fmv.d.x ft0, zero
    fsd  ft0, 0(t5)     # fzero = f0 (always 0 in RVV context)
    # Actually use a temp: store 0.0
    addi t5, t5, 8
    addi t6, t6, -1
    j    zero_out
zero_done:
    # accumulate: for each row i, out[j] += mat[i*cols+j]*vec[i]
    mv   t6, a3           # rows counter
    mv   t4, a1           # current row in mat
    mv   t3, a2           # vec pointer
matTvec_row:
    beqz t6, matTvec_done
    fld  ft2, 0(t3)       # scalar vec[i]
    fmv.d.x ft3, zero
    # broadcast ft2 into all lanes via vfmv would need scalar; use loop
    mv   t2, a4           # cols counter
    mv   t5, a0           # out pointer (reset per row)
    mv   t1, t4           # mat row pointer
col_loop:
    beqz t2, next_row
    fld  ft1, 0(t1)       # mat[i][j]
    fld  ft0, 0(t5)       # out[j]
    fmadd.d ft0, ft1, ft2, ft0   # out[j] += mat[i][j]*vec[i]
    fsd  ft0, 0(t5)
    addi t1, t1, 8
    addi t5, t5, 8
    addi t2, t2, -1
    j    col_loop
next_row:
    slli t5, a4, 3
    add  t4, t4, t5       # next row of mat
    addi t3, t3, 8        # next element of vec
    addi t6, t6, -1
    j    matTvec_row
matTvec_done:
    ret

# ==============================================================
# Helper: matmat_4x4(out, A, B)
#   4×4 matrix multiply: out = A * B
#   a0=out, a1=A, a2=B  (all 4×4 row-major doubles)
# ==============================================================
.globl matmat_4x4
matmat_4x4:
    addi sp, sp, -48
    sd   ra, 40(sp)
    sd   s0, 32(sp)
    sd   s1, 24(sp)
    sd   s2, 16(sp)
    sd   s3,  8(sp)
    sd   s4,  0(sp)

    mv   s0, a0   # out
    mv   s1, a1   # A
    mv   s2, a2   # B

    # For each row i=0..3, col j=0..3:
    #   out[i*4+j] = sum_k A[i*4+k]*B[k*4+j]
    li   s3, 0    # i
mm_row:
    li   t6, 4
    bge  s3, t6, mm_done
    li   s4, 0    # j
mm_col:
    bge  s4, t6, mm_col_done
    # compute dot product
    fmv.d.x ft0, zero
    li   a3, 0   # k
mm_k:
    li   a4, 4
    bge  a3, a4, mm_k_done
    # A[i*4+k]
    li   t0, 4
    mul  t0, s3, t0
    add  t0, t0, a3
    slli t0, t0, 3
    add  t0, t0, s1
    fld  ft1, 0(t0)
    # B[k*4+j]
    li   t1, 4
    mul  t1, a3, t1
    add  t1, t1, s4
    slli t1, t1, 3
    add  t1, t1, s2
    fld  ft2, 0(t1)
    fmadd.d ft0, ft1, ft2, ft0
    addi a3, a3, 1
    j    mm_k
mm_k_done:
    # store out[i*4+j]
    li   t0, 4
    mul  t0, s3, t0
    add  t0, t0, s4
    slli t0, t0, 3
    add  t0, t0, s0
    fsd  ft0, 0(t0)
    addi s4, s4, 1
    j    mm_col
mm_col_done:
    addi s3, s3, 1
    j    mm_row
mm_done:
    ld   s4,  0(sp)
    ld   s3,  8(sp)
    ld   s2, 16(sp)
    ld   s1, 24(sp)
    ld   s0, 32(sp)
    ld   ra, 40(sp)
    addi sp, sp, 48
    ret

# ==============================================================
# Helper: vec_add_N(out, a, b, N)  –  out = a + b  (N doubles)
# ==============================================================
.globl vec_add_N
vec_add_N:
va_loop:
    vsetvli t0, a3, e64, m1, ta, ma
    vle64.v v1, (a1)
    vle64.v v2, (a2)
    vfadd.vv v3, v1, v2
    vse64.v v3, (a0)
    slli t1, t0, 3
    add  a0, a0, t1
    add  a1, a1, t1
    add  a2, a2, t1
    sub  a3, a3, t0
    bnez a3, va_loop
    ret

# ==============================================================
# Helper: vec_sub_N(out, a, b, N)  –  out = a - b  (N doubles)
# ==============================================================
.globl vec_sub_N
vec_sub_N:
vs_loop:
    vsetvli t0, a3, e64, m1, ta, ma
    vle64.v v1, (a1)
    vle64.v v2, (a2)
    vfsub.vv v3, v1, v2
    vse64.v v3, (a0)
    slli t1, t0, 3
    add  a0, a0, t1
    add  a1, a1, t1
    add  a2, a2, t1
    sub  a3, a3, t0
    bnez a3, vs_loop
    ret

# ==============================================================
# Helper: mat_scale_add(out, eye4, scale, KH)
#   out[i] = eye4[i] - scale*KH[i]  for 16 elements
#   Actually: mat_sub_4x4(out, I, KH)
# ==============================================================
.globl mat_sub_4x4
mat_sub_4x4:
    # out = A - B, 16 doubles
    li   a3, 16
ms4_loop:
    vsetvli t0, a3, e64, m1, ta, ma
    vle64.v v1, (a1)
    vle64.v v2, (a2)
    vfsub.vv v3, v1, v2
    vse64.v v3, (a0)
    slli t1, t0, 3
    add a0, a0, t1
    add a1, a1, t1
    add a2, a2, t1
    sub a3, a3, t0
    bnez a3, ms4_loop
    ret

# ==============================================================
# Helper: invert2x2(inv, mat)
#   2×2 matrix inversion: inv = mat⁻¹
#   a0=inv (2×2), a1=mat (2×2)
# ==============================================================
.globl invert2x2
invert2x2:
    fld  ft0, 0(a1)    # a = mat[0][0]
    fld  ft1, 8(a1)    # b = mat[0][1]
    fld  ft2,16(a1)    # c = mat[1][0]
    fld  ft3,24(a1)    # d = mat[1][1]
    # det = a*d - b*c
    fmul.d ft4, ft0, ft3
    fmul.d ft5, ft1, ft2
    fsub.d ft6, ft4, ft5   # ft6 = det
    # 1/det
    la   t0, inv2x2_one
    fld  ft7, 0(t0)
    fdiv.d ft6, ft7, ft6   # ft6 = 1/det
    # inv = (1/det)*[[d,-b],[-c,a]]
    fmul.d ft0, ft3, ft6   # d/det
    fneg.d ft1, ft1
    fmul.d ft1, ft1, ft6   # -b/det
    fneg.d ft2, ft2
    fmul.d ft2, ft2, ft6   # -c/det
    fmul.d ft3, ft3, ft6   # a/det  -- wait reuse; fix:
    # redo properly with saved originals
    j inv2x2_proper

inv2x2_one: .double 1.0

inv2x2_proper:
    fld  fa0, 0(a1)    # a
    fld  fa1, 8(a1)    # b
    fld  fa2,16(a1)    # c
    fld  fa3,24(a1)    # d
    fmul.d ft4, fa0, fa3
    fmul.d ft5, fa1, fa2
    fsub.d ft6, ft4, ft5   # det
    la   t0, inv2x2_one2
    fld  ft7, 0(t0)
    fdiv.d ft6, ft7, ft6   # 1/det
    fmul.d ft0, fa3, ft6   # d/det
    fneg.d ft1, fa1
    fmul.d ft1, ft1, ft6   # -b/det
    fneg.d ft2, fa2
    fmul.d ft2, ft2, ft6   # -c/det
    fmul.d ft3, fa0, ft6   # a/det
    fsd  ft0,  0(a0)
    fsd  ft1,  8(a0)
    fsd  ft2, 16(a0)
    fsd  ft3, 24(a0)
    ret
inv2x2_one2: .double 1.0

# ==============================================================
# lkf_predict:
#   x_pred  = F * x
#   FP      = F * P          (4×4 matmat)
#   P_pred  = F * P * Fᵀ + Q  (4×4 matmat + add)
# ==============================================================
.section .text
.globl lkf_predict
lkf_predict:
    addi sp, sp, -48
    sd   ra, 40(sp)
    sd   s0, 32(sp)
    sd   s1, 24(sp)
    sd   s2, 16(sp)
    sd   s3,  8(sp)
    sd   s4,  0(sp)

    # --- x_pred = F * x  (4×4 mat times 4-vec) ---
    la   a0, lkf_Fx       # output
    la   a1, lkf_F        # matrix (4×4)
    la   a2, lkf_x        # vector (4)
    li   a3, 4            # rows
    li   a4, 4            # cols
    call matvec_NxM

    # copy Fx -> x
    la   a0, lkf_x
    la   a1, lkf_Fx
    li   a3, 4
cp_x:
    vsetvli t0, a3, e64, m1, ta, ma
    vle64.v v1, (a1)
    vse64.v v1, (a0)
    slli t1, t0, 3
    add a0,a0,t1; add a1,a1,t1
    sub a3,a3,t0
    bnez a3, cp_x

    # --- FP = F * P ---
    la   a0, lkf_FP
    la   a1, lkf_F
    la   a2, lkf_P
    call matmat_4x4

    # --- FPFt = FP * Fᵀ  (= FP * F transposed, so matmat_4x4 with Fᵀ)
    #  We compute col-by-col: for each col j of Fᵀ (= row j of F),
    #  do mat*(col), store. Actually just call matmat with B=F and take
    #  the transpose trick: FP*Fᵀ means for each element (i,j):
    #  sum_k FP[i][k]*F[j][k].
    #  Easiest: we have matmat_4x4 which does C=A*B.
    #  We need C = FP * Fᵀ.  Build Fᵀ in a scratch buffer then call.
    #  Use lkf_tmp16 as Fᵀ scratch.
    la   a0, lkf_tmp16     # destination: Fᵀ
    la   a1, lkf_F
    # transpose 4×4 manually
    li   t0, 0            # i
tp_i:
    li   t6, 4
    bge  t0, t6, tp_done
    li   t1, 0            # j
tp_j:
    bge  t1, t6, tp_j_done
    # src = F[i][j] = F + (i*4+j)*8
    li   t2, 4
    mul  t2, t0, t2
    add  t2, t2, t1
    slli t2, t2, 3
    add  t2, t2, a1
    fld  ft0, 0(t2)
    # dst = Ft[j][i] = tmp16 + (j*4+i)*8
    la   a0, lkf_tmp16
    li   t3, 4
    mul  t3, t1, t3
    add  t3, t3, t0
    slli t3, t3, 3
    add  t3, t3, a0
    fsd  ft0, 0(t3)
    addi t1, t1, 1
    j    tp_j
tp_j_done:
    addi t0, t0, 1
    j    tp_i
tp_done:
    # now FPFt = FP * Ft
    la   a0, lkf_FPFt
    la   a1, lkf_FP
    la   a2, lkf_tmp16    # Fᵀ
    call matmat_4x4

    # --- P = FPFt + Q ---
    la   a0, lkf_P
    la   a1, lkf_FPFt
    la   a2, lkf_Q
    li   a3, 16
    call vec_add_N

    ld   s4,  0(sp)
    ld   s3,  8(sp)
    ld   s2, 16(sp)
    ld   s1, 24(sp)
    ld   s0, 32(sp)
    ld   ra, 40(sp)
    addi sp, sp, 48
    ret

# ==============================================================
# lkf_update:
#   innov  = z - H*x
#   PHt    = P * Hᵀ          (4×4 times 4×2 → 4×2)
#   S      = H * P * Hᵀ + R  (2×2)
#   K      = PHt * S⁻¹       (4×2)
#   x     += K * innov
#   P      = (I - K*H) * P
#   Then write CSV line.
# ==============================================================
.section .data
.align 3
lkf_eye4:
    .double 1.0,0.0,0.0,0.0
    .double 0.0,1.0,0.0,0.0
    .double 0.0,0.0,1.0,0.0
    .double 0.0,0.0,0.0,1.0

lkf_fmt_str:
    .asciz "%lld,%.6f,%.6f,%.6f,%.6f\n"

.section .text
.globl lkf_update
lkf_update:
    addi sp, sp, -80
    sd   ra, 72(sp)
    sd   s0, 64(sp)
    sd   s1, 56(sp)
    sd   s2, 48(sp)
    sd   s3, 40(sp)
    sd   s4, 32(sp)
    sd   s5, 24(sp)
    sd   s6, 16(sp)
    sd   s7,  8(sp)
    sd   s8,  0(sp)

    # ---- Hx = H * x  (2×4 * 4 = 2) ----
    la   a0, lkf_Hx
    la   a1, lkf_H
    la   a2, lkf_x
    li   a3, 2         # rows of H
    li   a4, 4         # cols of H
    call matvec_NxM

    # ---- innov = z - Hx ----
    la   a0, lkf_innov
    la   a1, lkf_z
    la   a2, lkf_Hx
    li   a3, 2
    call vec_sub_N

    # ---- PHt = P * Hᵀ  (4×4 * 4×2 → 4×2)
    #  Compute column by column: PHt[:,j] = P * Ht[:,j] = P * H[j,:]ᵀ
    #  H is 2×4, Hᵀ is 4×2. Column j of Hᵀ = row j of H.
    la   s0, lkf_PHt
    la   s1, lkf_P     # 4×4
    la   s2, lkf_H     # 2×4: row0 → Ht col0, row1 → Ht col1
    li   s3, 0         # column index j
pht_col:
    li   t6, 2
    bge  s3, t6, pht_done
    # Extract column j of Hᵀ  = row j of H into a 4-element scratch
    # H row j starts at H + j*4*8
    li   t0, 4
    mul  t0, s3, t0
    slli t0, t0, 3
    add  t0, t0, s2    # pointer to H row j (4 doubles)
    # store that 4-vec as Ht_col in lkf_tmp4
    la   a0, lkf_tmp4
    mv   a1, t0
    li   a3, 4
cp_htcol:
    vsetvli t1, a3, e64, m1, ta, ma
    vle64.v v1, (a1)
    vse64.v v1, (a0)
    slli t2, t1, 3
    add a0,a0,t2; add a1,a1,t2
    sub a3,a3,t1; bnez a3, cp_htcol

    # PHt[:,j] = P * lkf_tmp4  (4×4 mat × 4-vec)
    # output goes to lkf_PHt row-interleaved: PHt[i][j] = PHt_base + (i*2+j)*8
    # Compute P*Ht_col into a temp 4-vec, then scatter
    la   a6, lkf_tmp4    # input vec (Ht col)
    # re-use lkf_Fx as temp output
    la   a0, lkf_Fx
    la   a1, lkf_P
    la   a2, lkf_tmp4
    li   a3, 4
    li   a4, 4
    call matvec_NxM

    # scatter result (4-vec at lkf_Fx) into PHt column s3
    la   a0, lkf_PHt
    la   a1, lkf_Fx
    li   t0, 0          # row index i
pht_scatter:
    li   t6, 4
    bge  t0, t6, pht_scatter_done
    # PHt[i][j] = PHt_base + (i*2 + j)*8
    li   t1, 2
    mul  t1, t0, t1
    add  t1, t1, s3
    slli t1, t1, 3
    add  t1, t1, a0
    # value at lkf_Fx[i]
    slli t2, t0, 3
    add  t2, t2, a1
    fld  ft0, 0(t2)
    fsd  ft0, 0(t1)
    addi t0, t0, 1
    j    pht_scatter
pht_scatter_done:
    addi s3, s3, 1
    j    pht_col
pht_done:

    # ---- S = H * PHt + R  (2×2)
    #  H is 2×4, PHt is 4×2. H*PHt → 2×2
    #  Compute element by element
    la   s4, lkf_S
    li   t0, 0   # i
s_i:
    li   t6, 2
    bge  t0, t6, s_i_done
    li   t1, 0   # j
s_j:
    bge  t1, t6, s_j_done
    fmv.d.x ft0, zero
    li   t2, 0   # k
s_k:
    li   t5, 4
    bge  t2, t5, s_k_done
    # H[i][k]: H + (i*4+k)*8
    li   t3, 4; mul t3,t0,t3; add t3,t3,t2; slli t3,t3,3
    la   a0, lkf_H; add t3,t3,a0; fld ft1,0(t3)
    # PHt[k][j]: PHt + (k*2+j)*8
    li   t4, 2; mul t4,t2,t4; add t4,t4,t1; slli t4,t4,3
    la   a0, lkf_PHt; add t4,t4,a0; fld ft2,0(t4)
    fmadd.d ft0,ft1,ft2,ft0
    addi t2,t2,1; j s_k
s_k_done:
    # add R[i][j]
    li   t3, 2; mul t3,t0,t3; add t3,t3,t1; slli t3,t3,3
    la   a0, lkf_R; add t3,t3,a0; fld ft1,0(t3)
    fadd.d ft0,ft0,ft1
    # store S[i][j]
    li   t3, 2; mul t3,t0,t3; add t3,t3,t1; slli t3,t3,3
    add t3,t3,s4; fsd ft0,0(t3)
    addi t1,t1,1; j s_j
s_j_done:
    addi t0,t0,1; j s_i
s_i_done:

    # ---- Sinv = S⁻¹ ----
    la   a0, lkf_Sinv
    la   a1, lkf_S
    call invert2x2

    # ---- K = PHt * Sinv  (4×2 * 2×2 → 4×2) ----
    la   s5, lkf_K
    li   t0, 0   # i (0..3)
k_i:
    li   t6, 4
    bge  t0, t6, k_done
    li   t1, 0   # j (0..1)
k_j:
    li   t6, 2
    bge  t1, t6, k_j_done
    fmv.d.x ft0, zero
    li   t2, 0   # k (0..1)
k_k:
    li   t5, 2
    bge  t2, t5, k_k_done
    # PHt[i][k]: PHt + (i*2+k)*8
    li   t3,2; mul t3,t0,t3; add t3,t3,t2; slli t3,t3,3
    la   a0,lkf_PHt; add t3,t3,a0; fld ft1,0(t3)
    # Sinv[k][j]: Sinv + (k*2+j)*8
    li   t4,2; mul t4,t2,t4; add t4,t4,t1; slli t4,t4,3
    la   a0,lkf_Sinv; add t4,t4,a0; fld ft2,0(t4)
    fmadd.d ft0,ft1,ft2,ft0
    addi t2,t2,1; j k_k
k_k_done:
    li   t3,2; mul t3,t0,t3; add t3,t3,t1; slli t3,t3,3
    add  t3,t3,s5; fsd ft0,0(t3)
    addi t1,t1,1; j k_j
k_j_done:
    addi t0,t0,1; j k_i
k_done:

    # ---- Kv = K * innov  (4×2 * 2 → 4) ----
    la   a0, lkf_Kv
    la   a1, lkf_K
    la   a2, lkf_innov
    li   a3, 4    # rows
    li   a4, 2    # cols
    call matvec_NxM

    # ---- x = x + Kv ----
    la   a0, lkf_x
    la   a1, lkf_x
    la   a2, lkf_Kv
    li   a3, 4
    call vec_add_N

    # ---- KH = K * H  (4×2 * 2×4 → 4×4) ----
    la   s6, lkf_KH
    li   t0,0   # i
kh_i:
    li   t6,4; bge t0,t6,kh_done
    li   t1,0   # j
kh_j:
    li   t6,4; bge t1,t6,kh_j_done
    fmv.d.x ft0,zero
    li   t2,0   # k
kh_k:
    li   t5,2; bge t2,t5,kh_k_done
    # K[i][k]: K + (i*2+k)*8
    li   t3,2; mul t3,t0,t3; add t3,t3,t2; slli t3,t3,3
    la   a0,lkf_K; add t3,t3,a0; fld ft1,0(t3)
    # H[k][j]: H + (k*4+j)*8
    li   t4,4; mul t4,t2,t4; add t4,t4,t1; slli t4,t4,3
    la   a0,lkf_H; add t4,t4,a0; fld ft2,0(t4)
    fmadd.d ft0,ft1,ft2,ft0
    addi t2,t2,1; j kh_k
kh_k_done:
    li   t3,4; mul t3,t0,t3; add t3,t3,t1; slli t3,t3,3
    add  t3,t3,s6; fsd ft0,0(t3)
    addi t1,t1,1; j kh_j
kh_j_done:
    addi t0,t0,1; j kh_i
kh_done:

    # ---- IKH = I4 - KH ----
    la   a0, lkf_IKH
    la   a1, lkf_eye4
    la   a2, lkf_KH
    call mat_sub_4x4

    # ---- P = IKH * P ----
    la   a0, lkf_tmp16
    la   a1, lkf_IKH
    la   a2, lkf_P
    call matmat_4x4
    # copy tmp16 -> P (16 doubles)
    la   a0, lkf_P
    la   a1, lkf_tmp16
    li   a3, 16
cp_p:
    vsetvli t0,a3,e64,m1,ta,ma
    vle64.v v1,(a1); vse64.v v1,(a0)
    slli t1,t0,3; add a0,a0,t1; add a1,a1,t1
    sub a3,a3,t0; bnez a3,cp_p

    # ---- Update simulated measurement (walk z forward) ----
    la   t0, lkf_z
    fld  ft0, 0(t0)
    fld  ft1, 8(t0)
    # load 0.1 constant
    la   t1, lkf_dt_val
    fld  ft2, 0(t1)
    fadd.d ft0, ft0, ft2
    fadd.d ft1, ft1, ft2
    fsd  ft0, 0(t0)
    fsd  ft1, 8(t0)

    # ---- Write CSV line ----
    # Open file on first frame
    la   t0, lkf_fptr
    ld   t1, 0(t0)
    bnez t1, lkf_write_line

    # fopen("lkf_output.csv","w")
    la   a0, lkf_csv_name
    la   a1, lkf_open_mode
    call fopen
    la   t0, lkf_fptr
    sd   a0, 0(t0)
    # Write header
    # a0 already has FILE* from fopen
    mv   s8, a0           # save fp
    la   a0, lkf_hdr     # string first (fputs arg0)
    mv   a1, s8           # FILE* second (fputs arg1)
    call fputs
    mv   a0, s8

lkf_write_line:
    la   t0, lkf_fptr
    ld   s8, 0(t0)        # FILE*

    # fprintf(fp, fmt, frame, x[0], x[1], x[2], x[3])
    mv   a0, s8
    la   a1, lkf_fmt_str
    la   t0, lkf_frame
    ld   a2, 0(t0)
    la   t0, lkf_x
    fld  ft0,  0(t0); fmv.x.d a3, ft0   # x[0] -> int reg (variadic ABI)
    fld  ft0,  8(t0); fmv.x.d a4, ft0   # x[1]
    fld  ft0, 16(t0); fmv.x.d a5, ft0   # x[2]
    fld  ft0, 24(t0); fmv.x.d a6, ft0   # x[3]
    call fprintf

    # increment frame
    la   t0, lkf_frame
    ld   t1, 0(t0)
    addi t1, t1, 1
    sd   t1, 0(t0)

    # Flush every 100 frames
    andi t2, t1, 0x7F
    bnez t2, lkf_no_flush
    mv   a0, s8
    call fflush
lkf_no_flush:

    ld   s8,  0(sp)
    ld   s7,  8(sp)
    ld   s6, 16(sp)
    ld   s5, 24(sp)
    ld   s4, 32(sp)
    ld   s3, 40(sp)
    ld   s2, 48(sp)
    ld   s1, 56(sp)
    ld   s0, 64(sp)
    ld   ra, 72(sp)
    addi sp, sp, 80
    ret

.section .data
.align 3
lkf_dt_val:    .double 0.1
lkf_csv_name:  .asciz "lkf_output.csv"
lkf_open_mode: .asciz "w"
lkf_hdr:       .asciz "frame,px,py,vx,vy\n"

# ==============================================================
# Legacy entry points kept for Makefile compatibility
# ==============================================================
.section .text
.globl matvec_vec
matvec_vec:
    # Signature: a0=out, a1=mat(NxN), a2=vec, a3=N
    mv   a4, a3
    j    matvec_NxM

.globl mat_add_vec
mat_add_vec:
    mv   a3, a3   # a3=N already
    j    vec_add_N

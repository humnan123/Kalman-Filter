# ==============================================================
#  ekf_vector.s  –  Extended Kalman Filter (RISC-V RV64GCV)
#
#  State: [px, py, vx, vy]  (N=4)
#  Measurement model: range + bearing  (nonlinear h(x))
#    z[0] = sqrt(px² + py²)         range
#    z[1] = atan2(py, px)            bearing
#  Jacobian H(x) (2×4):
#    H[0][0] = px/r,  H[0][1] = py/r,  H[0][2]=0, H[0][3]=0
#    H[1][0] = -py/r², H[1][1] = px/r², H[1][2]=0, H[1][3]=0
#  Prediction same as LKF (linear motion model).
# ==============================================================

.section .data
.align 3

.globl ekf_x
ekf_x:   .double 10.0, 10.0, 1.0, 0.5   # initial state

.globl ekf_P
ekf_P:
    .double 1.0,0.0,0.0,0.0
    .double 0.0,1.0,0.0,0.0
    .double 0.0,0.0,1.0,0.0
    .double 0.0,0.0,0.0,1.0

# F same as LKF (dt=0.1)
.globl ekf_F
ekf_F:
    .double 1.0,0.0,0.1,0.0
    .double 0.0,1.0,0.0,0.1
    .double 0.0,0.0,1.0,0.0
    .double 0.0,0.0,0.0,1.0

.globl ekf_Q
ekf_Q:
    .double 1e-4,0.0,0.0,0.0
    .double 0.0,1e-4,0.0,0.0
    .double 0.0,0.0,1e-2,0.0
    .double 0.0,0.0,0.0,1e-2

# Measurement noise R (2×2): range std=0.5m, bearing std=0.01rad
.globl ekf_R
ekf_R:
    .double 0.25,0.0
    .double 0.0, 1e-4

# Simulated measurement z
.globl ekf_z
ekf_z:  .double 14.142135, 0.785398   # sqrt(200), pi/4

# True simulation state (separate from filter estimate)
.globl ekf_true
ekf_true: .double 10.0, 10.0, 1.0, 0.5   # same initial as ekf_x

# Linearised H (updated each update step)
.globl ekf_H
ekf_H:  .fill 8, 8, 0   # 2×4

# Scratch
.globl ekf_Fx
ekf_Fx:    .double 0.0,0.0,0.0,0.0
.globl ekf_FP
ekf_FP:    .fill 16, 8, 0
.globl ekf_FPFt
ekf_FPFt:  .fill 16, 8, 0
.globl ekf_tmp16
ekf_tmp16: .fill 16, 8, 0
.globl ekf_tmp4
ekf_tmp4:  .double 0.0,0.0,0.0,0.0
.globl ekf_hx
ekf_hx:    .double 0.0,0.0
.globl ekf_innov
ekf_innov: .double 0.0,0.0
.globl ekf_PHt
ekf_PHt:   .fill 8,  8, 0
.globl ekf_S
ekf_S:     .fill 4,  8, 0
.globl ekf_Sinv
ekf_Sinv:  .fill 4,  8, 0
.globl ekf_K
ekf_K:     .fill 8,  8, 0
.globl ekf_Kv
ekf_Kv:    .double 0.0,0.0,0.0,0.0
.globl ekf_KH
ekf_KH:    .fill 16, 8, 0
.globl ekf_IKH
ekf_IKH:   .fill 16, 8, 0

.globl ekf_frame
ekf_frame: .quad 0
.globl ekf_fptr
ekf_fptr:  .quad 0

ekf_eye4:
    .double 1.0,0.0,0.0,0.0
    .double 0.0,1.0,0.0,0.0
    .double 0.0,0.0,1.0,0.0
    .double 0.0,0.0,0.0,1.0

ekf_one:   .double 1.0
ekf_dt:    .double 0.1
ekf_twopi: .double 6.283185307179586
ekf_pi:    .double 3.141592653589793
ekf_halfpi:.double 1.5707963267948966

ekf_csv_name:  .asciz "ekf_output.csv"
ekf_open_mode: .asciz "w"
ekf_hdr:       .asciz "frame,px,py,vx,vy\n"
ekf_fmt_str:   .asciz "%lld,%.6f,%.6f,%.6f,%.6f\n"

# ==============================================================
.section .text
.align 2

# ---- asm_atan2(double y, double x) -> double in fa0 ----
# Uses software approximation: atan2 via series
# Arguments: fa0=y, fa1=x   Return: fa0 = atan2(y,x)
.globl asm_atan2
asm_atan2:
    # We delegate to C libm via call: just tail-call atan2
    # (The assembler can call C functions freely)
    tail atan2

# ---- h_func_asm(hx_out, x_state) ----
# Computes nonlinear observation: hx[0]=range, hx[1]=bearing
# a0 = output (2 doubles), a1 = state x (4 doubles)
.globl h_func_asm
h_func_asm:
    addi sp, sp, -16
    sd   ra, 8(sp)

    fld  ft0, 0(a1)    # px
    fld  ft1, 8(a1)    # py
    # range = sqrt(px²+py²)
    fmul.d ft2, ft0, ft0
    fmul.d ft3, ft1, ft1
    fadd.d ft4, ft2, ft3
    fsqrt.d ft5, ft4          # range
    fsd  ft5, 0(a0)

    # bearing = atan2(py, px)  – call asm_atan2
    sd   a0, 0(sp)            # save output ptr
    fmv.d fa0, ft1            # y = py
    fmv.d fa1, ft0            # x = px
    call asm_atan2            # fa0 = atan2(py,px)
    ld   a0, 0(sp)
    fsd  fa0, 8(a0)

    ld   ra, 8(sp)
    addi sp, sp, 16
    ret

# ---- compute_H_jacobian(H_out, x_state) ----
# Linearise h around current state.
# H[0][:] = [ px/r,  py/r, 0, 0 ]
# H[1][:] = [-py/r², px/r², 0, 0 ]
compute_H_jacobian:
    fld  ft0, 0(a1)    # px
    fld  ft1, 8(a1)    # py
    fmul.d ft2, ft0, ft0
    fmul.d ft3, ft1, ft1
    fadd.d ft4, ft2, ft3  # r²
    fsqrt.d ft5, ft4      # r
    # Guard: if r < 1e-9 set to 1e-9
    la   t0, ekf_one
    fld  ft6, 0(t0)
    fdiv.d ft6, ft6, ft5  # 1/r
    fmul.d ft7, ft6, ft6  # 1/r²

    # H[0][0] = px/r = px*(1/r)
    fmul.d fa0, ft0, ft6
    fsd  fa0, 0(a0)
    # H[0][1] = py/r
    fmul.d fa1, ft1, ft6
    fsd  fa1, 8(a0)
    # H[0][2] = 0, H[0][3] = 0
    fmv.d.x fa2, zero
    fsd  fa2, 16(a0)
    fsd  fa2, 24(a0)

    # H[1][0] = -py/r²
    fneg.d fa3, ft1
    fmul.d fa3, fa3, ft7
    fsd  fa3, 32(a0)
    # H[1][1] = px/r²
    fmul.d fa4, ft0, ft7
    fsd  fa4, 40(a0)
    # H[1][2] = 0, H[1][3] = 0
    fsd  fa2, 48(a0)
    fsd  fa2, 56(a0)
    ret

# ==============================================================
# ekf_predict: identical prediction to LKF (linear model)
#   x = F*x,   P = F*P*Fᵀ + Q
# ==============================================================
.globl ekf_predict
ekf_predict:
    addi sp, sp, -48
    sd   ra, 40(sp)
    sd   s0, 32(sp); sd s1,24(sp); sd s2,16(sp)
    sd   s3,  8(sp); sd s4, 0(sp)

    # x = F*x
    la   a0, ekf_Fx
    la   a1, ekf_F
    la   a2, ekf_x
    li   a3, 4; li a4, 4
    call matvec_NxM
    # copy Fx -> x
    la a0, ekf_x; la a1, ekf_Fx; li a3,4
ekf_cp_x:
    vsetvli t0,a3,e64,m1,ta,ma
    vle64.v v1,(a1); vse64.v v1,(a0)
    slli t1,t0,3; add a0,a0,t1; add a1,a1,t1
    sub a3,a3,t0; bnez a3,ekf_cp_x

    # FP = F*P
    la a0,ekf_FP; la a1,ekf_F; la a2,ekf_P
    call matmat_4x4

    # Fᵀ into ekf_tmp16
    la   a0, ekf_tmp16
    la   a1, ekf_F
    li   t0,0
ekf_tp_i:
    li   t6,4; bge t0,t6,ekf_tp_done
    li   t1,0
ekf_tp_j:
    li   t6,4; bge t1,t6,ekf_tp_j_done
    li   t2,4; mul t2,t0,t2; add t2,t2,t1; slli t2,t2,3
    add  t2,t2,a1; fld ft0,0(t2)
    la   a0,ekf_tmp16
    li   t3,4; mul t3,t1,t3; add t3,t3,t0; slli t3,t3,3
    add  t3,t3,a0; fsd ft0,0(t3)
    addi t1,t1,1; j ekf_tp_j
ekf_tp_j_done:
    addi t0,t0,1; j ekf_tp_i
ekf_tp_done:

    # FPFt = FP*Fᵀ
    la a0,ekf_FPFt; la a1,ekf_FP; la a2,ekf_tmp16
    call matmat_4x4

    # P = FPFt + Q
    la a0,ekf_P; la a1,ekf_FPFt; la a2,ekf_Q; li a3,16
    call vec_add_N

    ld s4,0(sp); ld s3,8(sp); ld s2,16(sp)
    ld s1,24(sp); ld s0,32(sp); ld ra,40(sp)
    addi sp,sp,48
    ret

# ==============================================================
# ekf_update:
#   Compute H = jacobian of h at current x
#   hx = h(x)
#   innov = z - hx
#   PHt, S=H*PHt+R, K=PHt*Sinv
#   x += K*innov
#   P = (I-KH)*P
#   Write CSV
# ==============================================================
.globl ekf_update
ekf_update:
    addi sp, sp, -80
    sd   ra, 72(sp); sd s0,64(sp); sd s1,56(sp)
    sd   s2, 48(sp); sd s3,40(sp); sd s4,32(sp)
    sd   s5, 24(sp); sd s6,16(sp); sd s7,8(sp); sd s8,0(sp)

    # ---- Compute H Jacobian ----
    la   a0, ekf_H
    la   a1, ekf_x
    call compute_H_jacobian

    # ---- hx = h(x) ----
    la   a0, ekf_hx
    la   a1, ekf_x
    call h_func_asm

    # ---- innov = z - hx ----
    la   a0, ekf_innov
    la   a1, ekf_z
    la   a2, ekf_hx
    li   a3, 2
    call vec_sub_N

    # ---- PHt = P * Hᵀ  (4×4 * 4×2 → 4×2, col-by-col) ----
    la   s0, ekf_PHt
    la   s1, ekf_P
    la   s2, ekf_H
    li   s3, 0
ekf_pht_col:
    li   t6,2; bge s3,t6,ekf_pht_done
    li   t0,4; mul t0,s3,t0; slli t0,t0,3; add t0,t0,s2
    la   a0, ekf_tmp4; mv a1,t0; li a3,4
ekf_cp_htcol:
    vsetvli t1,a3,e64,m1,ta,ma
    vle64.v v1,(a1); vse64.v v1,(a0)
    slli t2,t1,3; add a0,a0,t2; add a1,a1,t2
    sub a3,a3,t1; bnez a3,ekf_cp_htcol
    la a0,ekf_Fx; la a1,ekf_P; la a2,ekf_tmp4; li a3,4; li a4,4
    call matvec_NxM
    la a0,ekf_PHt; la a1,ekf_Fx; li t0,0
ekf_pht_scat:
    li t6,4; bge t0,t6,ekf_pht_scat_done
    li t1,2; mul t1,t0,t1; add t1,t1,s3; slli t1,t1,3
    add t1,t1,a0
    slli t2,t0,3; add t2,t2,a1
    fld ft0,0(t2); fsd ft0,0(t1)
    addi t0,t0,1; j ekf_pht_scat
ekf_pht_scat_done:
    addi s3,s3,1; j ekf_pht_col
ekf_pht_done:

    # ---- S = H*PHt + R ----
    la s4, ekf_S
    li t0,0
ekf_s_i:
    li t6,2; bge t0,t6,ekf_s_i_done
    li t1,0
ekf_s_j:
    li t6,2; bge t1,t6,ekf_s_j_done
    fmv.d.x ft0,zero
    li t2,0
ekf_s_k:
    li t5,4; bge t2,t5,ekf_s_k_done
    li t3,4; mul t3,t0,t3; add t3,t3,t2; slli t3,t3,3
    la a0,ekf_H; add t3,t3,a0; fld ft1,0(t3)
    li t4,2; mul t4,t2,t4; add t4,t4,t1; slli t4,t4,3
    la a0,ekf_PHt; add t4,t4,a0; fld ft2,0(t4)
    fmadd.d ft0,ft1,ft2,ft0
    addi t2,t2,1; j ekf_s_k
ekf_s_k_done:
    li t3,2; mul t3,t0,t3; add t3,t3,t1; slli t3,t3,3
    la a0,ekf_R; add t3,t3,a0; fld ft1,0(t3)
    fadd.d ft0,ft0,ft1
    li t3,2; mul t3,t0,t3; add t3,t3,t1; slli t3,t3,3
    add t3,t3,s4; fsd ft0,0(t3)
    addi t1,t1,1; j ekf_s_j
ekf_s_j_done:
    addi t0,t0,1; j ekf_s_i
ekf_s_i_done:

    # ---- Sinv ----
    la a0,ekf_Sinv; la a1,ekf_S
    call invert2x2

    # ---- K = PHt * Sinv ----
    la s5, ekf_K
    li t0,0
ekf_k_i:
    li t6,4; bge t0,t6,ekf_k_done
    li t1,0
ekf_k_j:
    li t6,2; bge t1,t6,ekf_k_j_done
    fmv.d.x ft0,zero; li t2,0
ekf_k_k:
    li t5,2; bge t2,t5,ekf_k_k_done
    li t3,2; mul t3,t0,t3; add t3,t3,t2; slli t3,t3,3
    la a0,ekf_PHt; add t3,t3,a0; fld ft1,0(t3)
    li t4,2; mul t4,t2,t4; add t4,t4,t1; slli t4,t4,3
    la a0,ekf_Sinv; add t4,t4,a0; fld ft2,0(t4)
    fmadd.d ft0,ft1,ft2,ft0
    addi t2,t2,1; j ekf_k_k
ekf_k_k_done:
    li t3,2; mul t3,t0,t3; add t3,t3,t1; slli t3,t3,3
    add t3,t3,s5; fsd ft0,0(t3)
    addi t1,t1,1; j ekf_k_j
ekf_k_j_done:
    addi t0,t0,1; j ekf_k_i
ekf_k_done:

    # ---- Kv = K * innov ----
    la a0,ekf_Kv; la a1,ekf_K; la a2,ekf_innov; li a3,4; li a4,2
    call matvec_NxM

    # ---- x = x + Kv ----
    la a0,ekf_x; la a1,ekf_x; la a2,ekf_Kv; li a3,4
    call vec_add_N

    # ---- KH = K*H ----
    la s6, ekf_KH
    li t0,0
ekf_kh_i:
    li t6,4; bge t0,t6,ekf_kh_done
    li t1,0
ekf_kh_j:
    li t6,4; bge t1,t6,ekf_kh_j_done
    fmv.d.x ft0,zero; li t2,0
ekf_kh_k:
    li t5,2; bge t2,t5,ekf_kh_k_done
    li t3,2; mul t3,t0,t3; add t3,t3,t2; slli t3,t3,3
    la a0,ekf_K; add t3,t3,a0; fld ft1,0(t3)
    li t4,4; mul t4,t2,t4; add t4,t4,t1; slli t4,t4,3
    la a0,ekf_H; add t4,t4,a0; fld ft2,0(t4)
    fmadd.d ft0,ft1,ft2,ft0
    addi t2,t2,1; j ekf_kh_k
ekf_kh_k_done:
    li t3,4; mul t3,t0,t3; add t3,t3,t1; slli t3,t3,3
    add t3,t3,s6; fsd ft0,0(t3)
    addi t1,t1,1; j ekf_kh_j
ekf_kh_j_done:
    addi t0,t0,1; j ekf_kh_i
ekf_kh_done:

    # ---- IKH = I4 - KH ----
    la a0,ekf_IKH; la a1,ekf_eye4; la a2,ekf_KH
    call mat_sub_4x4

    # ---- P = IKH * P ----
    la a0,ekf_tmp16; la a1,ekf_IKH; la a2,ekf_P
    call matmat_4x4
    la a0,ekf_P; la a1,ekf_tmp16; li a3,16
ekf_cp_p:
    vsetvli t0,a3,e64,m1,ta,ma
    vle64.v v1,(a1); vse64.v v1,(a0)
    slli t1,t0,3; add a0,a0,t1; add a1,a1,t1
    sub a3,a3,t0; bnez a3,ekf_cp_p

    # ---- Advance true state: true = F * true  (dt=0.1) ----
    la   t0, ekf_true
    fld  ft0,  0(t0)   # true_px
    fld  ft1,  8(t0)   # true_py
    fld  ft2, 16(t0)   # true_vx
    fld  ft3, 24(t0)   # true_vy
    la   t2, ekf_dt
    fld  ft4, 0(t2)    # dt = 0.1
    fmadd.d ft0, ft2, ft4, ft0   # px += vx*dt
    fmadd.d ft1, ft3, ft4, ft1   # py += vy*dt
    fsd  ft0,  0(t0)
    fsd  ft1,  8(t0)
    # ---- Walk z from TRUE state ----
    la   t1, ekf_z
    fmul.d ft5,ft0,ft0; fmul.d ft6,ft1,ft1
    fadd.d ft5,ft5,ft6; fsqrt.d ft5,ft5
    fsd  ft5, 0(t1)    # z[0] = true range
    addi sp,sp,-16; sd ra,8(sp)
    fmv.d fa0,ft1; fmv.d fa1,ft0
    call asm_atan2
    ld   ra,8(sp); addi sp,sp,16
    la   t1,ekf_z; fsd fa0,8(t1)  # z[1] = true bearing

    # ---- CSV output ----
    la   t0, ekf_fptr; ld t1,0(t0); bnez t1, ekf_write_line
    la   a0, ekf_csv_name; la a1, ekf_open_mode
    call fopen
    la   t0, ekf_fptr; sd a0,0(t0)
    mv   s8, a0
    la   a0, ekf_hdr        # string first
    mv   a1, s8             # FILE* second
    call fputs
    mv   a0, s8
ekf_write_line:
    la t0,ekf_fptr; ld s8,0(t0)
    mv  a0,s8; la a1,ekf_fmt_str
    la  t0,ekf_frame; ld a2,0(t0)
    la  t0,ekf_x
    fld  ft0,  0(t0); fmv.x.d a3, ft0   # x[0] variadic->int reg
    fld  ft0,  8(t0); fmv.x.d a4, ft0   # x[1]
    fld  ft0, 16(t0); fmv.x.d a5, ft0   # x[2]
    fld  ft0, 24(t0); fmv.x.d a6, ft0   # x[3]
    call fprintf
    la  t0,ekf_frame; ld t1,0(t0); addi t1,t1,1; sd t1,0(t0)
    andi t2,t1,0x7F; bnez t2,ekf_no_flush
    mv a0,s8; call fflush
ekf_no_flush:
    ld s8,0(sp); ld s7,8(sp); ld s6,16(sp); ld s5,24(sp)
    ld s4,32(sp); ld s3,40(sp); ld s2,48(sp)
    ld s1,56(sp); ld s0,64(sp); ld ra,72(sp)
    addi sp,sp,80
    ret

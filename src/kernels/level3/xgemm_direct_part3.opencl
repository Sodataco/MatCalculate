
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 3 of 3 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

//改
// 定义 CEIL_DIV 宏
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// Vectorized type definitions
#if VWMD == 1
  typedef real realmvec;
#elif VWMD == 2
  typedef real2 realmvec;
#elif VWMD == 4
  typedef real4 realmvec;
#elif VWMD == 8
  typedef real8 realmvec;
#endif

#if VWND == 1
  typedef real realnvec;
#elif VWND == 2
  typedef real2 realnvec;
#elif VWND == 4
  typedef real4 realnvec;
#elif VWND == 8
  typedef real8 realnvec;
#endif

#if VWCD == 1
  typedef real realcvec;
#elif VWCD == 2
  typedef real2 realcvec;
#elif VWCD == 4
  typedef real4 realcvec;
#endif
//改end

/*原
// Main body of the kernel. This is the direct version without pre/post processing and restrictions.
INLINE_FUNC void XgemmDirect(const int kSizeM, const int kSizeN, const int kSizeK,
                             const real_arg arg_alpha,
                             const real_arg arg_beta,
                             const __global realMD* restrict agm, const int a_offset, const int a_ld,
                             const __global realND* restrict bgm, const int b_offset, const int b_ld,
                             __global real* cgm, const int c_offset, const int c_ld,
                             LOCAL_PTR real* alm, LOCAL_PTR real* blm,
                             const int a_transpose, const int b_transpose, const int c_transpose,
                             const int a_conjugate, const int b_conjugate) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Extra pointers to scalar versions of global memory
  const __global real* restrict agms = (const __global real* restrict) agm;
  const __global real* restrict bgms = (const __global real* restrict) bgm;

  // Allocates workitem-private memory (registers)
  #pragma promote_to_registers
  real apd[MWID];
  #pragma promote_to_registers
  real bpd[NWID];
  #pragma promote_to_registers
  real cpd[NWID * MWID];

  // Initializes the accumulation registers
  #pragma unroll
  /*原
  for (int _mi = 0; _mi < MWID; _mi += 1) {
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      SetToZero(cpd[_ni * MWID + _mi]);
    }
  }*/

  //改
  #pragma unroll
  for (int _ni = 0; _ni < NWID; _ni += 1) {
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
          // 检查当前元素是否在矩阵边界内
          bool inside_bounds = ((idm + _mi) < kSizeM) && ((idn + _ni) < kSizeN);
          
          if (inside_bounds) {
              // 使用优化后的存储函数
              StoreResultsDirectOptimized(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn,
                                        alpha, beta, c_ld, c_offset, c_transpose);
          }
          // 边界外的元素不需要处理
      }
  }
  //改end
  

  // The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
  // processes only the main parts: output blocks of WGD by WGD.
  const int idm = get_local_id(0) * MWID + GetGroupID0() * WGD;
  const int idn = get_local_id(1) * NWID + GetGroupID1() * WGD;
  if ((idm < (kSizeM/WGD)*WGD) && (idn < (kSizeN/WGD)*WGD)) {

    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (kSizeK/WGD) * WGD; kwg += WGD) {

      // Loads data: off-chip --> local (matrix A and B)
      if (a_ld % VWMD == 0 && a_offset % VWMD == 0) {
        GlobalToLocalDirectA(agm, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
      }
      else {
        GlobalToLocalScalarA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
      }
      if (b_ld % VWND == 0 && b_offset % VWND == 0) {
        GlobalToLocalDirectB(bgm, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
      }
      else {
        GlobalToLocalScalarB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int _pit = 0; _pit < KWID; _pit += 1) {
          int kg = pwi + _pit;

          // Loads data: local --> private (matrix A and B)
          #pragma unroll
          for (int _mi = 0; _mi < MWID; _mi += 1) {
            apd[_mi] = LocalToPrivateDirectA(alm, _mi, kg, a_transpose);
          }
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            bpd[_ni] = LocalToPrivateDirectB(blm, _ni, kg, b_transpose);
          }

          // Performs the accumulation (Cpmd += Apmd * Bpmd)
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            #pragma unroll
            for (int _mi = 0; _mi < MWID; _mi += 1) {
              MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < kSizeK; ++kwg) {

      // Loads data: off-chip --> private (matrix A and B)
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        apd[_mi] = GlobalToPrivateDirectA(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate);
      }
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        bpd[_ni] = GlobalToPrivateDirectB(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate);
      }

      // Performs the accumulation (Cpmd += Apmd * Bpmd)
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        #pragma unroll
        for (int _mi = 0; _mi < MWID; _mi += 1) {
          MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
        }
      }
    }

    // Stores a tile of results and performs the multiplication with alpha and beta
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        StoreResultsDirect(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn,
                           alpha, beta, c_ld, c_offset, c_transpose);
      }
    }
  }
*/

//改
INLINE_FUNC void XgemmDirect(const int kSizeM, const int kSizeN, const int kSizeK,
                             const real_arg arg_alpha,
                             const real_arg arg_beta,//缩放因子
                             const __global realMD* restrict agm,//矩阵A的全局内存指针
                             const int a_offset, // 矩阵 A 在全局内存中的起始偏移量
                             const int a_ld,//矩阵 A 的 leading dimension
                             const __global realND* restrict bgm, const int b_offset, const int b_ld,
                             __global real* cgm, const int c_offset, const int c_ld,
                             LOCAL_PTR real* alm, LOCAL_PTR real* blm,
                             const int a_transpose, const int b_transpose, const int c_transpose,
                             const int a_conjugate, const int b_conjugate) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Extra pointers to scalar versions of global memory
  const __global real* restrict agms = (const __global real* restrict) agm;
  const __global real* restrict bgms = (const __global real* restrict) bgm;

  // Allocates workitem-private memory (registers) - vectorized versions
  #pragma promote_to_registers
  realmvec apd_vec[CEIL_DIV(MWID, VWMD)];
  #pragma promote_to_registers
  realnvec bpd_vec[CEIL_DIV(NWID, VWND)];
  #pragma promote_to_registers
  realcvec cpd_vec[CEIL_DIV(MWID, VWCD) * NWID];

  // Initializes the accumulation registers - vectorized
  #pragma unroll
  for (int _mi = 0; _mi < CEIL_DIV(MWID, VWCD); _mi++) {
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni++) {
      SetToZero(cpd_vec[_ni * CEIL_DIV(MWID, VWCD) + _mi]);
    }
  }

  // The faster version of GEMM with vectorization
  const int idm = get_local_id(0) * MWID + GetGroupID0() * WGD;
  const int idn = get_local_id(1) * NWID + GetGroupID1() * WGD;
  if ((idm < (kSizeM/WGD)*WGD) && (idn < (kSizeN/WGD)*WGD)) {

    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (kSizeK/WGD) * WGD; kwg += WGD) {

      // Vectorized loading: off-chip --> local (matrix A and B)
      if (a_ld % VWMD == 0 && a_offset % VWMD == 0) {
        GlobalToLocalDirectA(agm, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
      }
      else {
        GlobalToLocalScalarA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
      }
      if (b_ld % VWND == 0 && b_offset % VWND == 0) {
        GlobalToLocalDirectB(bgm, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
      }
      else {
        GlobalToLocalScalarB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      // Vectorized computation: unrolled by VEC_K_UNROLL
      for (int pwi = 0; pwi < WGD; pwi += VEC_K_UNROLL) {
        #pragma unroll
        for (int _pit = 0; _pit < VEC_K_UNROLL; _pit++) {
          int kg = pwi + _pit;
          if (kg < WGD) {
            
            // Vectorized load: local --> private (matrix A)
            #pragma unroll
            for (int _mi = 0; _mi < CEIL_DIV(MWID, VWMD); _mi++) {
              apd_vec[_mi] = LocalToPrivateVectorA(alm, _mi, kg, a_transpose);
            }
            
            // Vectorized load: local --> private (matrix B)
            #pragma unroll
            for (int _ni = 0; _ni < CEIL_DIV(NWID, VWND); _ni++) {
              bpd_vec[_ni] = LocalToPrivateVectorB(blm, _ni, kg, b_transpose);
            }
            
            // Vectorized multiply-add
            #pragma unroll
            for (int _ni = 0; _ni < CEIL_DIV(NWID, VWND); _ni++) {
              #pragma unroll
              for (int _mi = 0; _mi < CEIL_DIV(MWID, VWMD); _mi++) {
                
                // Expand vectors to scalars for computation
                #pragma unroll
                for (int _vni = 0; _vni < VWND; _vni++) {
                  #pragma unroll
                  for (int _vmi = 0; _vmi < VWMD; _vmi++) {
                    const real a_val = GetVectorElement(apd_vec[_mi], _vmi);
                    const real b_val = GetVectorElement(bpd_vec[_ni], _vni);
                    
                    // Calculate target vector and element index
                    const int total_mi = _mi * VWMD + _vmi;
                    const int total_ni = _ni * VWND + _vni;
                    const int vec_idx = total_ni * CEIL_DIV(MWID, VWCD) + total_mi / VWCD;
                    const int elem_idx = total_mi % VWCD;
                    
                    // Accumulate result
                    AddToVectorElement(&cpd_vec[vec_idx], elem_idx, a_val * b_val);
                  }
                }
              }
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < kSizeK; ++kwg) {

      // Scalar loading for remaining elements
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi++) {
        const real a_val = GlobalToPrivateDirectA(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate);
        
        // Accumulate directly to result vectors
        #pragma unroll
        for (int _ni = 0; _ni < NWID; _ni++) {
          const int vec_idx = _ni * CEIL_DIV(MWID, VWCD) + _mi / VWCD;
          const int elem_idx = _mi % VWCD;
          AddToVectorElement(&cpd_vec[vec_idx], elem_idx, 
                             a_val * GlobalToPrivateDirectB(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate));
        }
      }
    }

    // Vectorized store: private --> off-chip (matrix C)
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni++) {
      #pragma unroll
      for (int _mi = 0; _mi < CEIL_DIV(MWID, VWCD); _mi++) {
        const int vec_idx = _ni * CEIL_DIV(MWID, VWCD) + _mi;
        StoreResultsVector(cgm, cpd_vec[vec_idx], _mi * VWCD, _ni, idm, idn,
                           alpha, beta, c_ld, c_offset, c_transpose);
      }
    }
  }

  //改end

  // Simple but slower version for the parts on the edge (incomplete tiles in M and N-dimensions)
  else {

  #pragma promote_to_registers
  real apd[MWID];
  #pragma promote_to_registers
  real bpd[NWID];
  #pragma promote_to_registers
  real cpd[NWID * MWID];

    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (kSizeK/WGD) * WGD; kwg+=WGD) {

      // Loads data: off-chip --> local (matrix A and B)
      GlobalToLocalCheckedA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate, kSizeM, kSizeK);
      GlobalToLocalCheckedB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate, kSizeN, kSizeK);
      barrier(CLK_LOCAL_MEM_FENCE);

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int _pit = 0; _pit < KWID; _pit += 1) {
          int kg = pwi + _pit;

          // Loads data: local --> private (matrix A and B)
          #pragma unroll
          for (int _mi = 0; _mi < MWID; _mi += 1) {
            apd[_mi] = LocalToPrivateDirectA(alm, _mi, kg, a_transpose);
          }
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            bpd[_ni] = LocalToPrivateDirectB(blm, _ni, kg, b_transpose);
          }

          // Performs the accumulation (C += A * B)
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            #pragma unroll
            for (int _mi = 0; _mi < MWID; _mi += 1) {
              MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < kSizeK; ++kwg) {

      // Loads data: off-chip --> private (matrix A and B)
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        apd[_mi] = GlobalToPrivateCheckedA(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate, kSizeM);
      }
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        bpd[_ni] = GlobalToPrivateCheckedB(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate, kSizeN);
      }

      // Performs the accumulation (C += A * B)
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        #pragma unroll
        for (int _mi = 0; _mi < MWID; _mi += 1) {
          MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
        }
      }
    }

    // Stores a tile of results and performs the multiplication with alpha and beta
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        StoreResultsChecked(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn, kSizeM, kSizeN,
                            alpha, beta, c_ld, c_offset, c_transpose);
      }
    }
  }
}


//new
// Vector load/store helper functions
INLINE_FUNC real GetVectorElement(realmvec vec, int index) {
  #if VWMD == 1
    return vec;
  #elif VWMD == 2
    return (index == 0) ? vec.s0 : vec.s1;
  #elif VWMD == 4
    switch(index) {
      case 0: return vec.s0;
      case 1: return vec.s1;
      case 2: return vec.s2;
      case 3: return vec.s3;
    }
  #elif VWMD == 8
    switch(index) {
      case 0: return vec.s0;
      case 1: return vec.s1;
      case 2: return vec.s2;
      case 3: return vec.s3;
      case 4: return vec.s4;
      case 5: return vec.s5;
      case 6: return vec.s6;
      case 7: return vec.s7;
    }
  #endif
  return 0;
}

INLINE_FUNC void AddToVectorElement(realcvec* vec, int index, real value) {
  #if VWCD == 1
    *vec += value;
  #elif VWCD == 2
    if (index == 0) (*vec).s0 += value;
    else (*vec).s1 += value;
  #elif VWCD == 4
    switch(index) {
      case 0: (*vec).s0 += value; break;
      case 1: (*vec).s1 += value; break;
      case 2: (*vec).s2 += value; break;
      case 3: (*vec).s3 += value; break;
    }
  #endif
}

INLINE_FUNC void StoreResultsVector(__global real* cgm, realcvec c_vec, 
                                   int mi_base, int ni, 
                                   int idm, int idn,
                                   real alpha, real beta,
                                   int c_ld, int c_offset,
                                   int c_transpose) {
  const int col = idn + ni;
  const int row_base = idm + mi_base;
  
  if (c_transpose == 0) {
    // Non-transposed storage
    for (int i = 0; i < VWCD; i++) {
      const int row = row_base + i;
      const int index = (col * c_ld) + row + c_offset;
      const real new_value = alpha * GetVectorElement(c_vec, i) + beta * cgm[index];
      cgm[index] = new_value;
    }
  } else {
    // Transposed storage
    for (int i = 0; i < VWCD; i++) {
      const int row = row_base + i;
      const int index = (row * c_ld) + col + c_offset;
      const real new_value = alpha * GetVectorElement(c_vec, i) + beta * cgm[index];
      cgm[index] = new_value;
    }
  }
}

// Vectorized local memory access
INLINE_FUNC realmvec LocalToPrivateVectorA(LOCAL_PTR real* alm, int mi, int kg, int a_transpose) {
  const int local_index = kg * (WGD + PADA) + (get_local_id(0) * MWID + mi * VWMD);
  return vloada_half(local_index, (__local real*)alm);
}

INLINE_FUNC realnvec LocalToPrivateVectorB(LOCAL_PTR real* blm, int ni, int kg, int b_transpose) {
  const int local_index = kg * (WGD + PADB) + (get_local_id(1) * NWID + ni * VWND);
  return vloada_half(local_index, (__local real*)blm);
}
//new end

// =================================================================================================

// Direct version of the GEMM kernel with [A, B] = [non-transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XgemmDirectNN(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [non-transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XgemmDirectNT(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XgemmDirectTN(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XgemmDirectTT(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================

// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 4 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

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

INLINE_FUNC realmvec LocalToPrivateVectorA(LOCAL_PTR real* alm, int mi, int kg, int a_transpose) {
  const int local_index = kg * (WGD + PADA) + (get_local_id(0) * MWID + mi * VWMD);

  #if VWMD == 1
    return alm[local_index];
  #elif VWMD == 2
    return (realmvec)(alm[local_index], alm[local_index+1]);
  #elif VWMD == 4
    return (realmvec)(alm[local_index], alm[local_index+1], 
                      alm[local_index+2], alm[local_index+3]);
  #elif VWMD == 8
    return (realmvec)(alm[local_index], alm[local_index+1], 
                      alm[local_index+2], alm[local_index+3],
                      alm[local_index+4], alm[local_index+5],
                      alm[local_index+6], alm[local_index+7]);
  #endif
}

INLINE_FUNC realnvec LocalToPrivateVectorB(LOCAL_PTR real* blm, int ni, int kg, int b_transpose) {
  const int local_index = kg * (WGD + PADB) + (get_local_id(1) * NWID + ni * VWND);
  
  #if VWND == 1
    return blm[local_index];
  #elif VWND == 2
    return (realnvec)(blm[local_index], blm[local_index+1]);
  #elif VWND == 4
    return (realnvec)(blm[local_index], blm[local_index+1], 
                      blm[local_index+2], blm[local_index+3]);
  #elif VWND == 8
    return (realnvec)(blm[local_index], blm[local_index+1], 
                      blm[local_index+2], blm[local_index+3],
                      blm[local_index+4], blm[local_index+5],
                      blm[local_index+6], blm[local_index+7]);
  #endif
}

/*
// Vectorized local memory access
INLINE_FUNC realmvec LocalToPrivateVectorA(LOCAL_PTR real* alm, int mi, int kg, int a_transpose) {
  const int local_index = kg * (WGD + PADA) + (get_local_id(0) * MWID + mi * VWMD);
  return vloada_half(local_index, (__local real*)alm);
}

INLINE_FUNC realnvec LocalToPrivateVectorB(LOCAL_PTR real* blm, int ni, int kg, int b_transpose) {
  const int local_index = kg * (WGD + PADB) + (get_local_id(1) * NWID + ni * VWND);
  return vloada_half(local_index, (__local real*)blm);
}*/
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
  XgemmDire0ct(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);
}


    // =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================

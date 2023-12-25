__kernel void gemm_image(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c, int m, int n, int k) {
   const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

   int2 coord = (int2)(get_global_id(0), get_global_id(1));
   float sum = 0.0f;

   for (int i = 0; i < k; ++i) {
       float aVal = read_imagef(a, sampler, coord + (int2)(0, i)).x;
       float bVal = read_imagef(b, sampler, coord + (int2)(i, 0)).x;
       sum += aVal * bVal;
   }

   write_imagef(c, coord, (float4)(sum, 0.0f, 0.0f, 0.0f));
}

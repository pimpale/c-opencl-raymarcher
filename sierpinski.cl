float3 quat_mul_vec3(float3 v, float4 q) {
  /* from linmath.h */
  float3 t = 2 * cross(q.xyz,v);
  return v + q.w * t + cross(q.xyz, t);
}
unsigned int float_to_color(float hue) {
  unsigned int redcomp = 127*(1+sin(hue+1.6));
  unsigned int greencomp = 127*(1+sin(hue-0.5));
  unsigned int bluecomp = 127*(1+sin(hue+3.7));
  return (redcomp << 16) + (greencomp << 8) + bluecomp;
}

float sierpinski(float3 z) {
  float scale = 2.0f;
  float offset = 20.0f;
  float r;
  int n = 0;
  while (n < 10) {
     if(z.x+z.y<0) z.xy = -z.yx;
     if(z.x+z.z<0) z.xz = -z.zx;
     if(z.y+z.z<0) z.zy = -z.yz;
     z = z*scale - offset*(scale-1.0f);
     n++;
  }
  return (length(z)) * pow(scale, 0.0f-n - 1);
}

float distance_function(float3 loc) {
  return sierpinski(loc);
}

void kernel cast(
                 const unsigned int x_size,
                 const unsigned int y_size,
                 global unsigned int* framebuffer,
                 const float3 eye,
                 const float4 quat,
                 const float time
                ) {
  const float EPSILON = 0.1f;
  const float MAX_DEPTH = 100;
  const unsigned int MAX_ITER = 1000;
  const unsigned int x = get_global_id(0);
  const unsigned int y = get_global_id(1);
  unsigned int color = 0x000000;
  float3 ray_direction = (float3) {
       x - (x_size/2.0),
       y - (y_size/2.0),
       y_size
     };
  ray_direction = normalize(quat_mul_vec3(ray_direction, quat));
  float depth = 0;
  for(int i = 0; i < MAX_ITER; i++) {
    float distance = distance_function(eye + depth*ray_direction);
    if(distance < EPSILON) {
      color = float_to_color(i/10.0);
      break;
    }
    depth += distance;
    if(depth > MAX_DEPTH) {
      break;
    }
  }
  /* set array value */
  framebuffer[y*x_size + x] = color;
}

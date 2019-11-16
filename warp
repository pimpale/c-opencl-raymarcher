float3 quat_mul_vec3(float3 v, float4 q) {
  /* from linmath.h */
  float3 t = 2 * cross(q.xyz,v);
  return v + q.w * t + cross(q.xyz, t);
}
unsigned int float_to_color(float hue) {
  unsigned int redcomp = 127*(1+sin(hue+1.6));
  unsigned int greencomp = 127*(1+sin(hue-0.3));
  unsigned int bluecomp = 127*(1+sin(hue+3.7));
  return (redcomp << 16) + (greencomp << 8) + bluecomp;
}

bool inCube(float3 loc) {
  uint3 uloc = abs((int3) {loc.x+7, loc.y+7, loc.z+7});
  if(uloc.x % 41 < 5 && uloc.y % 41 < 5 && uloc.z %41 <5) {
    return true;
  }
  return false;
}

bool inSphere(float3 loc) {
  return length(loc) < 1.5;
}

float3 bend(float3 loc, float3 vec, float time) {
  //float3 holeloc = (float3) {cos(time/10),sin(time/10),0};
  //float dist = length(loc-holeloc);
  //float3 accel = normalize(holeloc-loc)/(dist*dist);
  // faster
  float3 accel = loc/pow(length(loc),4);
  return vec + accel;
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
  const unsigned int MAX_ITER = 500;
  const unsigned int x = get_global_id(0);
  const unsigned int y = get_global_id(1);
  unsigned int color = 0x000000;
  float3 ray_direction = (float3) {
       x - (x_size/2.0),
       y - (y_size/2.0),
       y_size
     };
  ray_direction = normalize(quat_mul_vec3(ray_direction, quat));
  float3 loc = eye;
  for(int i = 0; i < MAX_ITER; i++) {
    ray_direction = bend(loc, ray_direction, time);
    if(inCube(loc)) {
      color = float_to_color(i/100.0);
      break;
    } else if(inSphere(loc)) {
      color = 0x000000;
      break;
    }
    loc += ray_direction;
  }
  /* set array value */
  framebuffer[y*x_size + x] = color;
}

// Shader to calculate h0k and h0minusk.
export const h0kshader = `
#define PI 3.1415926535897932384626433832795
#define g 9.81

// N and L will be int, but for type conversion passed as float.
uniform float N;
uniform float L;
uniform float A;
uniform vec2 w;
uniform float V;
uniform float k_coeff;
uniform sampler2D noise1;
uniform sampler2D noise2;

//Adaptation of https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
vec2 box_muller() {

    vec2 texCoord = gl_FragCoord.xy / N;

    // gl_FragCoord for index
    float n0 = clamp(texture2D(noise1, texCoord).r, 0.001, 1.0);
    float n1 = clamp(texture2D(noise2, texCoord).r, 0.001, 1.0);

    float mag = sqrt(-2.0 * log(n1));
    float z0 = mag * cos(2.0 * PI * n0);
    float z1 = mag * sin(2.0 * PI * n0);

    return vec2(z0, z1);

}

void main() {
    vec2 x = gl_FragCoord.xy - N / 2.0;
    vec2 k = (2.0 * PI / L) * x;

    float p_L = (V*V) / g;
    float p_L2 = p_L * p_L;
    float k_len = length(k);
    float k_len2 = k_len * k_len;
    float k_len4 = k_len2 * k_len2;
    float k_d_w = dot(normalize(k), normalize(w));
    float k_d_w2 = pow(k_d_w, 6.0);

    float d = 0.00001;
    float l2 = p_L2 * d * d;

    float phk = A * exp(-1.0/(k_len2 * p_L2)) / k_len4 * k_d_w2 * exp(-k_len2 * l2);
    vec2 h0k = 1.0/sqrt(2.0) * box_muller() * sqrt(phk);

    gl_FragColor = vec4(h0k.xy, 0, 1);

}

`

// Calculates y hkt.
export const hktshader_y = `
#define PI 3.1415926535897932384626433832795
#define g 9.81

// N and L will be int, but for type conversion passed as float.
uniform float N;
uniform float L;
uniform float t;
uniform sampler2D h0k;
uniform sampler2D h0minusk;

struct complex {
    float re;
    float im;
};

complex c_add(complex a, complex b) {
    return complex(a.re + b.re, a.im + b.im);
}

complex c_mult(complex a, complex b) {
    return complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

complex c_conj(complex c) {
    return complex(c.re, -c.im);
}

complex c_exp(float v) {
    return complex(cos(v), sin(v));
}


void main() {

    vec2 x = gl_FragCoord.xy;
    vec2 k = (2.0 * PI / L) * x;
    float k_len = length(k);
    float w_k = sqrt(k_len * g);
    float w_k_t = w_k * t;

    vec2 v_h0k = texture2D(h0k, gl_FragCoord.xy / N).xy;
    complex c_h0k = complex(v_h0k.x, v_h0k.y);
    vec2 v_h0minusk = texture2D(h0minusk, gl_FragCoord.xy / N).xy;
    complex c_h0minusk = complex(v_h0minusk.x, v_h0minusk.y);

    complex hkt = c_add(c_mult(c_h0k, c_exp(w_k_t)), c_mult(c_h0minusk, c_exp(-w_k_t)));
    gl_FragColor = vec4(hkt.re, hkt.im, 0, 1);
    // gl_FragColor = vec4(1, 1, 1, 1);
}
`

// Calculates x hkt using y hkt.
export const hktshader_x = `
#define PI 3.1415926535897932384626433832795

uniform float N;
uniform float L;

struct complex {
    float re;
    float im;
};

complex c_add(complex a, complex b) {
    return complex(a.re + b.re, a.im + b.im);
}

complex c_mult(complex a, complex b) {
    return complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

void main() {
    vec2 x = gl_FragCoord.xy;
    vec2 k = (2.0 * PI / L) * x;
    float k_len = length(k);

    // y_hkt passed in automatically as uniform from gpucomputationrenderer.
    vec2 v_hkt_y = texture2D(y_hkt, gl_FragCoord.xy / N).xy;
    complex c_hkt_y = complex(v_hkt_y.x, v_hkt_y.y);
    complex dx = c_mult(c_hkt_y, complex(0.0, -k.x / k_len));

    gl_FragColor = vec4(dx.re, dx.im, 0, 1);
}
`

// Calculates z hkt using y hkt
export const hktshader_z = `
#define PI 3.1415926535897932384626433832795

uniform float N;
uniform float L;

struct complex {
    float re;
    float im;
};

complex c_add(complex a, complex b) {
    return complex(a.re + b.re, a.im + b.im);
}

complex c_mult(complex a, complex b) {
    return complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

void main() {
    vec2 x = gl_FragCoord.xy;
    vec2 k = (2.0 * PI / L) * x;
    float k_len = length(k);

    // y_hkt passed in automatically as uniform from gpucomputationrenderer.
    vec2 v_hkt_y = texture2D(y_hkt, gl_FragCoord.xy / N).xy;
    complex c_hkt_y = complex(v_hkt_y.x, v_hkt_y.y);
    complex dz = c_mult(c_hkt_y, complex(0.0, -k.y / k_len));

    gl_FragColor = vec4(dz.re, dz.im, 0, 1);
}
`

// Computes derivatives outlined in Tessendorf's paper (bugged at the moment.)
export const ocean_derivative_shader = `
#define PI 3.1415926535897932384626433832795

uniform float N;
uniform float L;
uniform sampler2D hkt;

struct complex {
    float re;
    float im;
};

complex c_mult(complex a, complex b) {
    return complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

complex c_exp(float v) {
    return complex(cos(v), sin(v));
}

void main() {

    vec2 x = gl_FragCoord.xy - 0.5;
    vec2 k = (2.0 * PI / L) * x;
    complex ik = c_mult(complex(0.0, 1.0), complex(k.x, k.y));
    vec2 v_hkt = texture2D(hkt, x / N).xy;

    complex exponential = c_exp(dot(k, x));
    complex coefficient = c_mult(ik, complex(v_hkt.x, v_hkt.y));

    complex norm = c_mult(coefficient, exponential);

    gl_FragColor = vec4(norm.re, norm.im, 0, 1);
}
`

// Computes the normals given derivatives (derivatives are bugged at the moment.)
export const ocean_normal_shader = `
uniform float N;
uniform sampler2D delx;
uniform sampler2D dely;
uniform sampler2D delz;

void main() {

    vec2 x = gl_FragCoord.xy / N;

    float dy_dx = texture2D(dely, x).x;
    float dy_dz = texture2D(dely, x).y;
    float dx_dx = texture2D(delx, x).x;
    float dz_dz = texture2D(delz, x).y;

    vec2 s = vec2(dy_dx / (1.0 + dx_dx), dy_dz / (1.0 + dz_dz));

    vec3 norm = normalize(vec3(-s.x, 1, -s.y));

    gl_FragColor = vec4(norm, 1.0);
}
`

// Generates butterfly texture for input size N given bit reversed indices.
export const generate_butterfly_shader = `
#define PI 3.1415926535897932384626433832795

uniform int[%INDICES_SIZE] fft_indices;
uniform float N;

struct complex {
    float re;
    float im;
};

complex c_add(complex a, complex b) {
    return complex(a.re + b.re, a.im + b.im);
}

complex c_mult(complex a, complex b) {
    return complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

complex c_exp(float v) {
    return complex(cos(v), sin(v));
}

void main() {
    vec2 x = gl_FragCoord.xy - 0.5;
    float k = mod(x.y * N/pow(2.0, x.x + 1.0), N);
    complex twiddle = c_exp( 2.0 * PI * k / N );

    int span = int(pow(2.0, x.x));

    int wing;

    if (mod(x.y, pow(2.0, x.x + 1.0)) < pow(2.0, x.x)) wing = 1;
    else wing = 0;

    vec4 value;
    if (x.x == 0.0) {
        if (wing == 1) value = vec4(twiddle.re, twiddle.im, fft_indices[int(x.y)], fft_indices[int(x.y + 1.0)]);
        else value = vec4(twiddle.re, twiddle.im, fft_indices[int(x.y - 1.0)], fft_indices[int(x.y)]);
    } else {
        if (wing == 1) value = vec4(twiddle.re, twiddle.im, x.y, x.y + float(span));
        else value = vec4(twiddle.re, twiddle.im, x.y - float(span), x.y);
    }

    gl_FragColor = value;
}
`

// Computes FFT incrementally via ping-pong.
export const fft_ping_pong_shader = `
// (read) ping -> pong (write)
#define PI 3.1415926535897932384626433832795

uniform float N;
uniform int stage;
uniform int direction;
uniform sampler2D butterfly;

struct complex {
    float re;
    float im;
};

complex c_add(complex a, complex b) {
    return complex(a.re + b.re, a.im + b.im);
}

complex c_mult(complex a, complex b) {
    return complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

void horizontal_pass() {
    vec2 x = gl_FragCoord.xy;
    vec2 butterfly_lookup = vec2(float(stage) / log2(N), x.x / N);
    vec4 twiddle = texture2D(butterfly, butterfly_lookup).rgba;
    vec2 sample_p = texture2D(%INPUT_TEXTURE, vec2(twiddle.z, x.y) / N).xy;
    vec2 sample_q = texture2D(%INPUT_TEXTURE, vec2(twiddle.w, x.y) / N).xy;
    
    complex p = complex(sample_p.x, sample_p.y);
    complex q = complex(sample_q.x, sample_q.y);
    complex w = complex(twiddle.x, twiddle.y);

    complex h = c_add(p, c_mult(q, w));

    gl_FragColor = vec4(h.re, h.im, 0, 1);
}

void vertical_pass() {
    vec2 x = gl_FragCoord.xy;
    vec2 butterfly_lookup = vec2(float(stage) / log2(N), x.y / N);
    vec4 twiddle = texture2D(butterfly, butterfly_lookup);
    vec2 sample_p = texture2D(%INPUT_TEXTURE, vec2(x.x, twiddle.z) / N).xy;
    vec2 sample_q = texture2D(%INPUT_TEXTURE, vec2(x.x, twiddle.w) / N).xy;
    
    complex p = complex(sample_p.x, sample_p.y);
    complex q = complex(sample_q.x, sample_q.y);
    complex w = complex(twiddle.x, twiddle.y);

    complex h = c_add(p, c_mult(q, w));

    gl_FragColor = vec4(h.re, h.im, 0, 1);
}

void main() {
    if (direction == 0) horizontal_pass();
    else vertical_pass();
}
`

// Inverses Cooley-Tukey fft.
export const ifft_shader = `
uniform sampler2D final_buffer;
uniform float N;

void main() {
    // TODO might need to round this to int?
    vec2 x = gl_FragCoord.xy - 0.5;
    float permutations[2] = float[2](1.0, -1.0);
    int idx = int(mod(x.x + x.y, 2.0));
    float permutation = permutations[idx];
    float h = texture2D(final_buffer, x / N).y;
    float inversion = permutation * (h/float(N*N));
    gl_FragColor = vec4(inversion, inversion, inversion, 1);
}
`

// Adds displacement to surface vertices and calculates normals via averaging neighbouring surface normals.
// Calculates light and halfway vectors needed for Blinn-Phong model.
export const ocean_displacement_vertex_shader = `

varying float vertical_bump_amt;
varying vec2 vUv;
uniform float N;
uniform sampler2D dx;
uniform sampler2D dy;
uniform sampler2D dz;
uniform sampler2D normalTexture;
uniform float bump_scale;

out vec4 bump;
out vec3 skybox_coord;
out vec3 light_vector;
out vec3 normal_vector;
out vec3 halfway_vector;
out float fog_amt;

const vec2 size = vec2(2.0,0.0);
const vec3 light_position = vec3(10.0,10.0,10.0);
// TODO make lambda uniform.
const float lambda = 1.0;

void main() {

    vUv = uv;

    float dx_ = texture2D(dx, uv).x;
    float dy_ = texture2D(dy, uv).x;
    float dz_ = texture2D(dz, uv).x;

    // Might want to omit this coefficient here for standard range?
    vertical_bump_amt = dy_ * bump_scale;

    vec3 off = vec3(-1.0,0.0,1.0) / N;

    float m = texture2D(dy, uv).x;
    float l = texture2D(dy, uv + off.xy).x;
    float r = texture2D(dy, uv + off.zy).x;
    float d = texture2D(dy, uv + off.yx).x;
    float u = texture2D(dy, uv + off.yz).x;
    vec3 va = normalize(vec3(size.xy,r-l));
    vec3 vb = normalize(vec3(size.yx,u-d));
    bump = vec4( cross(va,vb), m );

    vec3 newPosition  = position + vec3(lambda * dx_, dy_, lambda * dz_) * bump_scale;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition , 1.0); 
    fog_amt = min(-(modelViewMatrix * vec4(newPosition, 1.0)).z/2000.0, 1.0);

    vec4 v = modelViewMatrix * vec4(newPosition, 1.0);
    vec3 norm = normalize(bump.xyz);

    skybox_coord = (inverse(transpose(modelMatrix)) * vec4(norm, 0.0)).xyz;
    light_vector = normalize((viewMatrix * vec4(light_position, 1.0)).xyz - v.xyz);
    normal_vector = (inverse(transpose(modelViewMatrix)) * vec4(norm, 0.0)).xyz;
    halfway_vector = light_vector + normalize(-v.xyz);
}

`;

// Applies lighting using Blinn Phong model and fog.
export const ocean_fragment_shader = `

uniform samplerCube skybox;
varying float vertical_bump_amt;
in vec4 bump;
in vec3 skybox_coord;
in vec3 normal_vector;
in vec3 light_vector;
in vec3 halfway_vector;
in float fog_amt;

void main() {

    vec3 norm_vec = normalize(normal_vector);
    vec3 light_vec = normalize(light_vector);
    vec3 half_vec = normalize(halfway_vector);

    // We can choose to do some skybox reflections with another coefficient c, but i think it looks better without.
    vec4 c = vec4(1,1,1,1);
    // vec4 c = texture(skybox, skybox_coord);

    vec4 ambient_col  = vec4(0.0, 0.65, 0.75, 1.0);
    vec4 diffuse_col  = vec4(0.5, 0.65, 0.75, 1.0);
    vec4 specular_col = vec4(1.0, 0.25, 0.0,  1.0);

    float ambient_amt  = 0.30;
    float diffuse_amt  = 0.30;
    float specular_amt = 1.80;
 
    float d = dot(norm_vec, light_vec);
    bool facing = d > 0.0;
 
    gl_FragColor = ambient_col  * ambient_amt  * c +
            diffuse_col * diffuse_amt * c * max(d, 0.0) +
                    (d > 0.0 ?
            specular_col * specular_amt * c * max(pow(dot(norm_vec, half_vec), 80.0), 0.0001) :
            vec4(0.0, 0.0, 0.0, 0.0));
 
    gl_FragColor = gl_FragColor * (1.0-fog_amt) + vec4(0.25, 0.75, 0.65, 1.0) * (fog_amt);
    gl_FragColor.a = 1.0;

}
`

// WIP - A stockham FFT may be more suitable for computation here as no bit-reversed indices are required.
// Using psuedocode from the following paper:
// https://www.microsoft.com/en-us/research/publication/fast-computation-of-general-fourier-transforms-on-gpus
export const stockham_ifft = `
precision highp float;
#define PI 3.1415926535897932384626433832795

// uniform sampler2D u_input;
uniform float u_transformSize;
uniform float u_subtransformSize;
uniform float direction;

struct complex {
    float re;
    float im;
};

complex c_add(complex a, complex b) {
    return complex(a.re + b.re, a.im + b.im);
}

complex c_mult(complex a, complex b) {
    return complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

complex c_exp(float v) {
    return complex(cos(v), sin(v));
}

void main(void){

    float idx;
    if (direction == 0.0) idx = gl_FragCoord.x - 0.5;
    else idx = gl_FragCoord.y - 0.5;

    float base = floor(idx / u_subtransformSize) * (u_subtransformSize / 2.0);
    float offset = mod(idx, u_subtransformSize / 2.0);
    float x0 = base + offset;
    float x1 = x0 + u_transformSize / 2.0;

    vec2 v0;
    vec2 v1;
    if (direction == 0.0) {
        v0 = texture2D(u_input, vec2(x0 + 0.5, gl_FragCoord.y) / u_transformSize).xy;
        v1 = texture2D(u_input, vec2(x1 + 0.5, gl_FragCoord.y) / u_transformSize).xy;
    } else {
        v0 = texture2D(u_input, vec2(gl_FragCoord.x, x0) / u_transformSize).xy;
        v1 = texture2D(u_input, vec2(gl_FragCoord.x, x1) / u_transformSize).xy;
    }

    complex c0 = complex(v0.x, v0.y);
    complex c1 = complex(v1.x, v1.y);

    float angle = 2.0 * PI * (idx / u_subtransformSize);
    complex ct = c_exp(angle);

    complex result = c_add(c0, c_mult(ct, c1));

    gl_FragColor = vec4(result.re, result.im, 0, 1);
}
`
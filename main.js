import * as THREE from "https://threejsfundamentals.org/threejs/resources/threejs/r122/build/three.module.js";
import { OrbitControls } from "https://threejsfundamentals.org/threejs/resources/threejs/r122/examples/jsm/controls/OrbitControls.js";
import { GPUComputationRenderer } from "https://threejsfundamentals.org/threejs/resources/threejs/r122/examples/jsm/misc/GPUComputationRenderer.js";
import { 
    h0kshader, 
    hktshader_y, hktshader_x, hktshader_z, 
    generate_butterfly_shader, 
    fft_ping_pong_shader, ifft_shader, 
    ocean_displacement_vertex_shader,
    ocean_derivative_shader,
    ocean_normal_shader,
    ocean_fragment_shader
} from "./wave-shader.js";

"use strict";

// Random seeds for noise textures used in Box Muller transform.
const [RND_SEED_1, RND_SEED_2, RND_SEED_3, RND_SEED_4] = [42, 301, 222, 1550];

// Ocean parameters (Note: changing these without knowing what they do could cause unexpected results)
const NUM_SEGMENTS = 256;
const PATCH_WIDTH = 2000;
const WIND_SPEED = 40;
const WIND_DIRECTION = {x: 1, y: 0};
const AMPLITUDE = 250.0;

const PLANE_WIDTH = 1000;
const OCEAN_SPEED = 0.2;

let clock = new THREE.Clock();
let t = 1;
let deltaTime = 0;

// Init scene and controls.
const [scene, camera, renderer, skybox] = initScene();
const controls = new OrbitControls(camera, renderer.domElement);

// Generate precomputed textures.
const [h0k, h0minusk] = inith0ks();
const butterflyTex = initButterfly();

// Init FFT Computation renderer.
const [fftCompute, fftVariables] = initFFT();

// Init hkt Computation renderer.
const [gpuCompute, y_hktVariable, x_hktVariable, z_hktVariable] = inithktShaders(h0k, h0minusk);

// Init normal Computation renderer. (WIP - not fully working at the moment.)
const [normCompute, delxVariable, delyVariable, delzVariable, normVariable] = initNormalShaders();

const water = initWaterSurface();

addEventListeners();
animate();

function initScene() {
    const scene = new THREE.Scene();

    // Set up the camera, move it to (3, 4, 5) and look at the origin (0, 0, 0).
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 6000);
    camera.position.set(645, 101, -10);
    camera.lookAt(new THREE.Vector3(0, 0, 0));
    // Setting rotation order to YXZ allows for yawing, pitching and rolling by rotation on the specified axes. (Used in headlook mode)

    // Set up the Web GL renderer.
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio); // HiDPI/retina rendering
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // Basic point lighting.
    const lights = [];
    lights[0] = new THREE.PointLight(0xffffff, 1, 0);
    lights[1] = new THREE.PointLight(0xffffff, 1, 0);
    lights[2] = new THREE.PointLight(0xffffff, 1, 0);

    lights[0].position.set(0, 200, 0);
    lights[1].position.set(100, 200, 100);
    lights[2].position.set(-100, -200, -100);

    scene.add(lights[0]);
    scene.add(lights[1]);
    scene.add(lights[2]);
    scene.add(new THREE.AmbientLight(0x404040))

    const loader = new THREE.CubeTextureLoader();
    const texture = loader.load([
        'miramar_lf.png',
        'miramar_rt.png',
        'miramar_up.png',
        'miramar_dn.png',
        'miramar_ft.png',
        'miramar_bk.png',
    ]);
    scene.background = texture;

    return [scene, camera, renderer, texture]
}

/**
 * Replace target string in shader program. Somewhat hacky.
 * @param {*} shader Shader to replace string.
 * @param {*} keyword String to replace
 * @param {*} replacement Replacement string
 * @returns String with replacement.
 */
function replaceInShader(shader, keyword, replacement) {

    return shader.replaceAll(keyword, replacement);

}

/**
 * Helper func to create an FFT variable as all use the same shader to compute.
 */
function createFFTVariable(gpuCompute, varName) {

    let pingpong = gpuCompute.createTexture();
    let result = gpuCompute.createTexture();
    fillTextureWithZeros(pingpong);
    fillTextureWithZeros(result);

    let fftVariable = gpuCompute.addVariable( 
        varName, 
        replaceInShader(fft_ping_pong_shader, "%INPUT_TEXTURE", varName), 
        pingpong 
    );
    let ifftVariable = gpuCompute.addVariable( 'ifft', ifft_shader, result );

    gpuCompute.setVariableDependencies( fftVariable, [fftVariable] );
    gpuCompute.setVariableDependencies( ifftVariable, []);

    let fftUniforms = fftVariable.material.uniforms;
    fftUniforms['butterfly'] = { value: butterflyTex };
    fftUniforms['N'] = { value: NUM_SEGMENTS };

    let ifftUniforms = ifftVariable.material.uniforms;
    ifftUniforms['N'] = { value: NUM_SEGMENTS };

    return [fftVariable, ifftVariable];
}

function initFFT() {
    const gpuCompute = new GPUComputationRenderer(NUM_SEGMENTS,NUM_SEGMENTS,renderer);

    // Add variables for hkts and derivatives.
    let [x_fftVariable, x_ifftVariable] = createFFTVariable(gpuCompute, "x_fft");
    let [y_fftVariable, y_ifftVariable] = createFFTVariable(gpuCompute, "y_fft");
    let [z_fftVariable, z_ifftVariable] = createFFTVariable(gpuCompute, "z_fft");
    let [del_nx_fftVariable, del_nx_ifftVariable] = createFFTVariable(gpuCompute, "delx_fft");
    let [del_ny_fftVariable, del_ny_ifftVariable] = createFFTVariable(gpuCompute, "dely_fft");
    let [del_nz_fftVariable, del_nz_ifftVariable] = createFFTVariable(gpuCompute, "delz_fft");

    const err = gpuCompute.init();

    if ( err !== null ) console.log(err);
    else {

        return [
            gpuCompute, 
            [
                {fft: x_fftVariable, ifft: x_ifftVariable, axis: 0, type: 0},
                {fft: y_fftVariable, ifft: y_ifftVariable, axis: 1, type: 0},
                {fft: z_fftVariable, ifft: z_ifftVariable, axis: 2, type: 0},

                {fft: del_nx_fftVariable, ifft: del_nx_ifftVariable, axis: 0, type: 1},
                {fft: del_ny_fftVariable, ifft: del_ny_ifftVariable, axis: 1, type: 1},
                {fft: del_nz_fftVariable, ifft: del_nz_ifftVariable, axis: 2, type: 1}
            ]
        ];
    }

}

function computeFFT(x_hkt, y_hkt, z_hkt, delx, dely, delz) {

    // Copy the hkts to the current buffers to start the computation.
    let startTexture;
    for (let v of fftVariables) {
        if (v.type == 0) startTexture = v.axis == 0 ? x_hkt : (v.axis == 1 ? y_hkt : z_hkt)
        else startTexture = v.axis == 0 ? delx : (v.axis == 1 ? dely : delz)
        fftCompute.renderTexture(startTexture, fftCompute.getCurrentRenderTarget(v.fft));
        // Set direction uniform for horizontal pass.
        v.fft.material.uniforms['direction'] = { value: 0 };
    }

    // Horizontal pass.
    for (let i = 0; i < Math.log2(NUM_SEGMENTS); i++) {

        for (let v of fftVariables) {
            v.fft.material.uniforms['stage'] = { value: i };
        }
        // Do ping-pong by repeatedly calling compute().
        fftCompute.compute();
    }

    // Update direction for vertical pass
    for (let v of fftVariables) {
        v.fft.material.uniforms['direction'] = { value: 1 };
    }

    // Vertical pass.
    for (let j = 0; j < Math.log2(NUM_SEGMENTS); j++) {

        for (let v of fftVariables) {
            v.fft.material.uniforms['stage'] = { value: j };
        }

        fftCompute.compute();
    }

    // Get output of all variables.
    let tx = [];
    for (let v of fftVariables) {
        tx.push({ifft: v.ifft, texture: fftCompute.getCurrentRenderTarget(v.fft).texture});
    }

    return tx;
}

function computeIFFT(textures) {
    // Copy FFT textures to input buffer.
    for (let v of textures) {
        v.ifft.material.uniforms['final_buffer'] = { value: v.texture }
    }
    // Compute inverse.
    fftCompute.compute();

    // Get outputs.
    let tx = [];
    for (let v of textures) {
        tx.push(fftCompute.getCurrentRenderTarget(v.ifft).texture)
    }

    return tx;
}

/**
 * Helper function to compute IFFT of all inputs (expects displacements & derivatives in freq. domain)
 */
function computeDisplacementAndDerivatives(x_hkt, y_hkt, z_hkt, sx, sy, sz) {
    
    let textures = computeFFT(x_hkt, y_hkt, z_hkt, sx, sy, sz);
    return computeIFFT(textures);
}

function initButterfly() {
    const gpuCompute = new GPUComputationRenderer(parseInt(Math.log2(NUM_SEGMENTS)), NUM_SEGMENTS, renderer);
    let butterflyTexture = gpuCompute.createTexture();
    fillTextureWithZeros(butterflyTexture);

    let butterflyVariable = gpuCompute.addVariable( 
        'butterfly', 
        // Allow for different size by updating the size of the indices uniform (hacky.)
        replaceInShader(generate_butterfly_shader, "%INDICES_SIZE", NUM_SEGMENTS.toString()), 
        butterflyTexture
    );

    gpuCompute.setVariableDependencies( butterflyVariable, [butterflyVariable] );

    let butterflyUniforms = butterflyVariable.material.uniforms;

    butterflyUniforms['N'] = { value: NUM_SEGMENTS };
    butterflyUniforms['fft_indices'] = { value: getBitReversedIndices(NUM_SEGMENTS) }

    const err = gpuCompute.init();
    if ( err !== null ) console.log(err);
    else {
        gpuCompute.compute();
        return gpuCompute.getCurrentRenderTarget(butterflyVariable).texture;
    }
}

function inith0ks() {

    const gpuCompute = new GPUComputationRenderer(NUM_SEGMENTS, NUM_SEGMENTS, renderer);
    let h0k = gpuCompute.createTexture();
    let h0minusk = gpuCompute.createTexture();
    fillTextureWithZeros(h0k);
    fillTextureWithZeros(h0minusk);

    let h0kVariable = gpuCompute.addVariable( 'h0k', h0kshader, h0k );
    let h0minuskVariable = gpuCompute.addVariable( 'h0minusk', h0kshader, h0minusk );

    gpuCompute.setVariableDependencies( h0kVariable, [h0kVariable] );
    gpuCompute.setVariableDependencies( h0minuskVariable, [h0kVariable] );

    // Cannot reassign uniforms obj, not mutable for shaders; assign each key individually.
    let h0kUniforms = h0kVariable.material.uniforms;
    let h0minuskUniforms = h0minuskVariable.material.uniforms;

    let noise1 = gpuCompute.createTexture();
    let noise2 = gpuCompute.createTexture();
    let noise3 = gpuCompute.createTexture();
    let noise4 = gpuCompute.createTexture();
    fillTextureWithNoise(noise1, NUM_SEGMENTS, RND_SEED_1);
    fillTextureWithNoise(noise2, NUM_SEGMENTS, RND_SEED_2);
    fillTextureWithNoise(noise3, NUM_SEGMENTS, RND_SEED_3);
    fillTextureWithNoise(noise4, NUM_SEGMENTS, RND_SEED_4);

    h0kUniforms['N'] = { value: NUM_SEGMENTS };
    h0kUniforms['A'] = { value: AMPLITUDE };
    h0kUniforms['L'] = { value: PATCH_WIDTH };
    h0kUniforms['w'] = { value: new THREE.Vector2(WIND_DIRECTION.x, WIND_DIRECTION.y) };
    h0kUniforms['V'] = { value: WIND_SPEED };
    h0kUniforms['k_coeff'] = { value: 1.0 };
    h0kUniforms['noise1'] = { value: noise1 };
    h0kUniforms['noise2'] = { value: noise2 };

    h0minuskUniforms['N'] = { value: NUM_SEGMENTS };
    h0minuskUniforms['A'] = { value: AMPLITUDE };
    h0minuskUniforms['L'] = { value: PATCH_WIDTH };
    h0minuskUniforms['w'] = { value: new THREE.Vector2(WIND_DIRECTION.x, WIND_DIRECTION.y)  };
    h0minuskUniforms['V'] = { value: WIND_SPEED };
    h0kUniforms['k_coeff'] = { value: -1.0 }; // minus k.
    h0minuskUniforms['noise1'] = { value: noise3 };
    h0minuskUniforms['noise2'] = { value: noise4 };


    const err = gpuCompute.init();
    if ( err !== null ) console.log(err);
    else {
        gpuCompute.compute();
        return [gpuCompute.getCurrentRenderTarget(h0kVariable).texture, gpuCompute.getCurrentRenderTarget(h0minuskVariable).texture];
    }

}

function initNormalShaders() {
    
    const gpuCompute = new GPUComputationRenderer(NUM_SEGMENTS, NUM_SEGMENTS, renderer);
    // Might need to change render target here?
    let delx = gpuCompute.createTexture();
    let dely = gpuCompute.createTexture();
    let delz = gpuCompute.createTexture();
    let normTexture = gpuCompute.createTexture();
    fillTextureWithZeros(delx);
    fillTextureWithZeros(dely);
    fillTextureWithZeros(delz);
    fillTextureWithZeros(normTexture);

    let delxVariable = gpuCompute.addVariable( 'delx', ocean_derivative_shader, delx );
    let delyVariable = gpuCompute.addVariable( 'dely', ocean_derivative_shader, dely );
    let delzVariable = gpuCompute.addVariable( 'delz', ocean_derivative_shader, delz );
    let normVariable = gpuCompute.addVariable( 'normTexture', ocean_normal_shader, normTexture );

    // do y shader first, x and z depend on this calculation.
    gpuCompute.setVariableDependencies( delxVariable, [delxVariable, delyVariable, delzVariable] );
    gpuCompute.setVariableDependencies( delyVariable, [delxVariable, delyVariable, delzVariable] );
    gpuCompute.setVariableDependencies( delzVariable, [delxVariable, delyVariable, delzVariable] );
    gpuCompute.setVariableDependencies( normVariable, [normVariable] );

    // Cannot reassign uniforms obj, not mutable for shaders; assign each key individually.
    let delxUniforms = delxVariable.material.uniforms;
    delxUniforms['N'] = { value: NUM_SEGMENTS };
    delxUniforms['L'] = { value: PATCH_WIDTH };

    let delyUniforms = delyVariable.material.uniforms;
    delyUniforms['N'] = { value: NUM_SEGMENTS };
    delyUniforms['L'] = { value: PATCH_WIDTH };

    let delzUniforms = delzVariable.material.uniforms;
    delzUniforms['N'] = { value: NUM_SEGMENTS };
    delzUniforms['L'] = { value: PATCH_WIDTH };

    let normUniforms = normVariable.material.uniforms;
    normUniforms['N'] = { value: NUM_SEGMENTS };
    normUniforms['L'] = { value: PATCH_WIDTH };

    const err = gpuCompute.init();
    if ( err !== null ) console.log(err);
    else {
        gpuCompute.compute();
        return [gpuCompute, delxVariable, delyVariable, delzVariable, normVariable];
    }
}

function computeDerivativesSpectrum(x_hkt, y_hkt, z_hkt) {
    delxVariable.material.uniforms['hkt'] = { value: x_hkt };
    delyVariable.material.uniforms['hkt'] = { value: y_hkt };
    delzVariable.material.uniforms['hkt'] = { value: z_hkt };

    normCompute.compute();

    return [
        normCompute.getCurrentRenderTarget(delxVariable).texture,
        normCompute.getCurrentRenderTarget(delyVariable).texture,
        normCompute.getCurrentRenderTarget(delzVariable).texture,
    ]
}

function computeNormals(delx, dely, delz) {

    normVariable.material.uniforms['delx'] = { value: delx };
    normVariable.material.uniforms['dely'] = { value: dely };
    normVariable.material.uniforms['delz'] = { value: delz };

    normCompute.compute();

    return normCompute.getCurrentRenderTarget(normVariable).texture;

}

function inithktShaders(h0k, h0minusk) {

    const hktCompute = new GPUComputationRenderer(NUM_SEGMENTS, NUM_SEGMENTS, renderer);
    let hkt_y = hktCompute.createTexture();
    let hkt_x = hktCompute.createTexture();
    let hkt_z = hktCompute.createTexture();
    fillTextureWithZeros(hkt_y);
    fillTextureWithZeros(hkt_x);
    fillTextureWithZeros(hkt_z);

    let y_hktVariable = hktCompute.addVariable( 'y_hkt', hktshader_y, hkt_y );
    let x_hktVariable = hktCompute.addVariable( 'x_hkt', hktshader_x, hkt_x );
    let z_hktVariable = hktCompute.addVariable( 'z_hkt', hktshader_z, hkt_z );

    // do y shader first, x and z depend on this calculation.
    hktCompute.setVariableDependencies( y_hktVariable, [y_hktVariable] );
    hktCompute.setVariableDependencies( x_hktVariable, [x_hktVariable, z_hktVariable, y_hktVariable] );
    hktCompute.setVariableDependencies( z_hktVariable, [x_hktVariable, z_hktVariable, y_hktVariable] );

    let y_hktUniforms = y_hktVariable.material.uniforms;
    y_hktUniforms['N'] = { value: NUM_SEGMENTS };
    y_hktUniforms['L'] = { value: PATCH_WIDTH };
    y_hktUniforms['t'] = { value: 0.0 };
    y_hktUniforms['h0k'] = { value: h0k };
    y_hktUniforms['h0minusk'] = { value: h0minusk };

    let x_hktUniforms = x_hktVariable.material.uniforms;
    x_hktUniforms['N'] = { value: NUM_SEGMENTS };
    x_hktUniforms['L'] = { value: PATCH_WIDTH };

    let z_hktUniforms = z_hktVariable.material.uniforms;
    z_hktUniforms['N'] = { value: NUM_SEGMENTS };
    z_hktUniforms['L'] = { value: PATCH_WIDTH };

    const err = hktCompute.init();
    if ( err !== null ) console.log(err);
    else {
        hktCompute.compute();
        return [hktCompute, y_hktVariable, x_hktVariable, z_hktVariable];
    }

}

function computehkt(gpuCompute, y_hktVariable, x_hktVariable, z_hktVariable, t) {

    y_hktVariable.material.uniforms['t'] = { value: t };
    gpuCompute.compute();

    return [
        gpuCompute.getCurrentRenderTarget(y_hktVariable).texture,
        gpuCompute.getCurrentRenderTarget(x_hktVariable).texture, 
        gpuCompute.getCurrentRenderTarget(z_hktVariable).texture, 
    ]

}

/**
 * Bit-reverses a sequence of numbers up to n for use in Cooley-Tukey FFT.
 * i.e. 0001 becomes 1000 (1 becomes 8).
 */
function getBitReversedIndices(n) {
    let exponent = parseInt(Math.log2(n))
    let arr = new Array(n);

    for (let i = 0; i < n; i++) {
        let s = i.toString(2);
        arr[i] = parseInt(s.padStart(exponent, '0').split('').reverse().join(''), 2);
    }

    return arr
}

function fillTextureWithZeros(texture) {
    let arr = texture.image.data;

    for (let i = 0; i < arr.length; i++) {
        arr[i] = 0;
    }
}

function fillTextureWithNoise(texture) {
    let arr = texture.image.data;

    let rnd;
    for (let i = 0; i < arr.length; i+=4) {
        rnd = Math.random();
        arr[i] = rnd
        arr[i+1] = rnd
        arr[i+2] = rnd
        arr[i+3] = rnd
    }
}

function initWaterSurface() {

    var surface = new THREE.PlaneGeometry(PLANE_WIDTH,PLANE_WIDTH,NUM_SEGMENTS,NUM_SEGMENTS);
    // Align with x-z plane.
    surface.rotateX(THREE.Math.degToRad(-90));
    // Add shader materials for displacement, lighting and fog.
    var surfaceMat = new THREE.ShaderMaterial({
        uniforms: {
            N: {value: NUM_SEGMENTS},
            dx: {value: null}, 
            dy: {value: null},
            dz: {value: null},
            normalTexture: {value: null},
            bump_scale: {value: 1.0},
            skybox: {value: skybox}
        },
        vertexShader: ocean_displacement_vertex_shader,
        fragmentShader: ocean_fragment_shader,
        transparent: true,
        side: THREE.DoubleSide
    })

    // Make multiple planes for larger ocean. Position side-by-side.
    const STITCH_TRANSLATION = 10; // Account for some stitching - not perfect but makes it less severe.
    const ROWS = 3;
    const COLS = 3;
    for (let i = -ROWS; i < ROWS; i++) {
        for (let j = -COLS; j < COLS; j++) {
            var surfaceMesh = new THREE.Mesh(surface, surfaceMat);
            surfaceMesh.position.set(PLANE_WIDTH * -i + (i * STITCH_TRANSLATION), 0, PLANE_WIDTH * -j + (j * STITCH_TRANSLATION))
            scene.add(surfaceMesh);
        }
    }

    return surfaceMesh;
}

function animate() {
    requestAnimationFrame(animate);

    deltaTime = clock.getDelta();

    // Compute hkts
    let [y_hkt, x_hkt, z_hkt] = computehkt(gpuCompute, y_hktVariable, x_hktVariable, z_hktVariable, t);

    // Compute derivatives in freq. domain (WIP - bugged at the moment)
    let [sx, sy, sz] = computeDerivativesSpectrum(x_hkt, y_hkt, z_hkt);

    // Compute displacements and derivatives via IFFT2.
    let [dx, dy, dz, delx, dely, delz] = computeDisplacementAndDerivatives(x_hkt, y_hkt, z_hkt, sx, sy, sz);

    // Compute normals via derivative (WIP - bugged at the moment).
    let normals = computeNormals(delx, dely, delz);

    // Update water material.
    water.material.uniforms['dx'].value = dx;
    water.material.uniforms['dy'].value = dy;
    water.material.uniforms['dz'].value = dz;
    water.material.uniforms['normalTexture'].value = normals;

    // Update t parameter based on time passed.
    t += OCEAN_SPEED * deltaTime;
    
    renderer.render(scene, camera);
}

function addEventListeners() {
    window.addEventListener('resize', () => {

        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);

    }, false);
}
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RectAreaLightUniformsLib } from "three/addons/lights/RectAreaLightUniformsLib.js";

// ------------------------------------------------------------- streamlit
// A component has to announce itself or Streamlit leaves the frame blank.
// Height is reported as 0 because theme.css takes the iframe out of the page
// flow entirely and sizes it itself; if that CSS ever fails to apply we would
// rather have nothing than a giant empty box shoved into the layout.
function post(type, data) {
  parent.postMessage({ isStreamlitMessage: true, type, ...data }, "*");
}
post("streamlit:componentReady", { apiVersion: 1 });
post("streamlit:setFrameHeight", { height: 0 });

// --------------------------------------------------------------- settings
const EXPOSURE = 0.16;
const SHADOW = 96; // percent
const TILT_DEG = 22;
const SPINS = 8; // whole turns per loop
const PERIOD = 400.0; // seconds
const WOBBLE = 4.65; // 465%
const FRICTION = 1.1;

// ---------------------------------------------------------------- renderer
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
} catch {
  // No WebGL. Leave the corner empty rather than throwing; the page is fine
  // without its decoration.
  throw new Error("webgl unavailable");
}
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x000000, 0); // real transparency, not a matte
renderer.toneMapping = THREE.AgXToneMapping; // same view transform as the Blender grade
renderer.toneMappingExposure = EXPOSURE;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();

// Blender: cam at (0, -7.6r, 2.2r) looking at the ball, which puts it 7.91
// units out. The normalisation below leaves the ball at radius 0.577, so it
// subtends 4.18 deg and a 24 deg lens would render it at a third of the frame -
// fine for the test rig, too small here, where theme.css sizes the iframe
// assuming the ball fills 60% of it. Tightening the lens rather than moving the
// camera keeps the distance-to-radius ratio, so the foreshortening is unchanged
// and this is a pure crop of the view the settings were dialled in against.
const camera = new THREE.PerspectiveCamera(13.9, 1, 0.1, 500);
camera.position.set(0, 2.2, 7.6);
camera.lookAt(0, 0, 0);

// ------------------------------------------------------------------ lights
// Ported from the Blender rig. Blender area-light watts and three.js
// RectAreaLight nits are different units, so the ratios are preserved and the
// absolute level is carried by the exposure above.
RectAreaLightUniformsLib.init();

const D = 20; // Blender D = radius * 20, and radius is 1 here
const LIGHT_SCALE = 9.0;

function flood(energy, size, color, pos) {
  const l = new THREE.RectAreaLight(
    new THREE.Color(...color),
    energy * LIGHT_SCALE,
    size,
    size,
  );
  l.position.set(pos[0] * D, pos[1] * D, pos[2] * D);
  l.lookAt(0, 0, 0);
  scene.add(l);
  return l;
}

// Blender is Z-up, three.js is Y-up: (x, y, z)_blender -> (x, z, -y)_three
const keyLight = flood(33.0, 5.7, [1.0, 0.72, 0.41], [0.8, 1.05, 0.4]);
const rimLight = flood(30.0, 5.1, [1.0, 0.63, 0.33], [-0.45, 0.75, -1.1]);
const fillLight = flood(2.6, 15.7, [1.0, 0.66, 0.38], [-0.85, 0.2, 0.5]);
const underLight = flood(0.9, 15.7, [0.7, 0.79, 1.0], [0.1, -0.9, -0.2]);

// Only a trace of ambient - the Blender world is a near-black night sky, and
// anything more than this flattens the floodlight modelling.
const hemi = new THREE.HemisphereLight(0x2a3550, 0x0a0c12, 0.09);
scene.add(hemi);

// "Shadow" here is the terminator, not a cast shadow: three.js RectAreaLights
// cannot cast shadows at all, and a lone sphere has nothing to cast onto
// anyway. What darkens the ball is the key-to-everything-else ratio, so this
// holds the key up and pulls the rim, fill, underlight and ambient down.
{
  const s = SHADOW / 100;
  rimLight.intensity *= 1 - 0.88 * s; // the rim is what fills the dark side
  fillLight.intensity *= 1 - 0.92 * s;
  underLight.intensity *= 1 - 0.92 * s;
  hemi.intensity *= 1 - 0.92 * s;
  keyLight.intensity *= 1 + 0.45 * s; // keep the lit side at level
}

// --------------------------------------------------------------- animation
// spinner: turns about the ball's own axis, SPINS times per loop
// tilter:  the off-axis tilt plus a slow wobble, so the tumble isn't straight
// flinger: what the pointer drives, about the world axes, outside the tilt
const flinger = new THREE.Group();
const tilter = new THREE.Group();
const spinner = new THREE.Group();
flinger.add(tilter);
tilter.add(spinner);
scene.add(flinger);

const tilt = THREE.MathUtils.degToRad(TILT_DEG);
const WOB_X = THREE.MathUtils.degToRad(7);
const WOB_Z = THREE.MathUtils.degToRad(9);

// The loop stays seamless as long as both the spin and the wobble complete a
// WHOLE number of cycles per period. Spin runs at SPINS turns, the wobble stays
// at one. Because they run at different rates the axis is somewhere new on each
// turn, so the orientation keeps changing instead of repeating.
function poseAt(time) {
  const p = (time % PERIOD) / PERIOD; // 0..1, exactly periodic
  spinner.rotation.y = p * Math.PI * 2 * SPINS;
  const wob = p * Math.PI * 2;
  tilter.rotation.x = tilt + WOBBLE * WOB_X * Math.sin(wob);
  tilter.rotation.z = WOBBLE * WOB_Z * Math.sin(wob + Math.PI / 3);
}

// --------------------------------------------------------------- flywheel
// Scrolling the page kicks the ball, layered ON TOP of the looping spin above,
// so however hard it gets kicked the underlying loop is untouched - let it coast
// back to rest and the animation is still perfectly seamless.
// Angular velocity is a VECTOR, not a pair of angles: its direction is the axis
// the ball turns about and its length is how fast. Building separate yaw and
// pitch terms and decaying them independently is what makes an off-axis spin
// gimbal instead of holding one honest axis.
const flyVec = new THREE.Vector3(); // rad/s, world space
const _axis = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _right = new THREE.Vector3();
const _up = new THREE.Vector3();
const _fwd = new THREE.Vector3();
const _kick = new THREE.Vector3();

// Rotate about a world axis by pre-multiplying, so the turn lands in world space
// rather than in whatever frame the ball has already accumulated.
function spinAbout(vec, angleScale) {
  const len = vec.length();
  if (len < 1e-9) return;
  _axis.copy(vec).divideScalar(len);
  _q.setFromAxisAngle(_axis, len * angleScale);
  flinger.quaternion.premultiply(_q);
}

function integrateFling(dt) {
  // exponential decay: the same fraction of speed is lost per unit time, which
  // is how a real flywheel with bearing drag actually coasts down
  flyVec.multiplyScalar(Math.exp(-FRICTION * dt));
  if (flyVec.length() < 0.01) {
    flyVec.set(0, 0, 0);
    return;
  }
  spinAbout(flyVec, dt);
}

// ----------------------------------------------------------------- scroll
// The ball sits behind the page content, so it never sees a pointer - the page
// scroll is what drives it instead. Scrolling down rolls it away from you and
// scrolling up rolls it back, and the harder the scroll the faster it goes.
const RAD_PER_PX = 0.022; // rad/s of spin per pixel scrolled
const MAX_VEL = 90; // ~14 rev/s, past which it just strobes
const BURST_MS = 220; // gap that counts as a new scroll gesture

// Semi-random: the axis is mostly the roll you would expect from the scroll
// direction, tipped by a pair of weights that are redrawn once per gesture. So
// one continuous scroll spins about one steady axis - it does not judder - but
// no two scrolls tumble it the same way.
let tipA = 0,
  tipB = 0,
  lastKickT = -1e9;

function scrollKick(dpx) {
  const now = performance.now();
  if (now - lastKickT > BURST_MS) {
    tipA = (Math.random() * 2 - 1) * 0.7; // sideways lean
    tipB = (Math.random() * 2 - 1) * 0.45; // twist about the view axis
  }
  lastKickT = now;

  camera.matrixWorld.extractBasis(_right, _up, _fwd);
  const mag = dpx * RAD_PER_PX;
  _kick
    .copy(_right)
    .multiplyScalar(mag)
    .addScaledVector(_up, mag * tipA)
    .addScaledVector(_fwd, mag * tipB);

  flyVec.add(_kick);
  if (flyVec.length() > MAX_VEL) flyVec.setLength(MAX_VEL);
}

// A flick with no scroll behind it: a random axis in the plane of the screen,
// tipped a little towards the viewer. Uniform over the whole sphere would let it
// pinwheel flat-on now and then, which reads as a glitch rather than a spin.
function randomSpin() {
  camera.matrixWorld.extractBasis(_right, _up, _fwd);
  const a = Math.random() * Math.PI * 2;
  _kick
    .copy(_right)
    .multiplyScalar(Math.cos(a))
    .addScaledVector(_up, Math.sin(a))
    .addScaledVector(_fwd, (Math.random() * 2 - 1) * 0.35)
    .normalize();
  flyVec.addScaledVector(_kick, 14 + Math.random() * 16);
  if (flyVec.length() > MAX_VEL) flyVec.setLength(MAX_VEL);
}

// Streamlit hands the component its arguments in a render message. `spin` is
// whichever game is on screen; when it changes, a new one has been pulled up and
// the ball gets flicked. The first render only records the value - otherwise
// every page load would open with a spin nobody asked for.
let lastSpin = null;
addEventListener("message", (e) => {
  if (e.data?.type !== "streamlit:render") return;
  const spin = e.data.args?.spin ?? "";
  if (lastSpin !== null && spin !== lastSpin) randomSpin();
  lastSpin = spin;
});

// Streamlit scrolls an element inside the host page, not the window, and the
// component is served from the app's own origin - so this is a plain same-origin
// scroll listener, in capture because scroll events do not bubble. If the host
// ever ends up on another origin this throws and the ball just keeps looping.
try {
  const host = parent.document;
  const seen = new WeakMap(); // per element, so two scrollers cannot swap
  host.addEventListener(
    "scroll",
    (e) => {
      // deltas and fake a huge jump
      const el = e.target === host ? host.documentElement : e.target;
      const top = el.scrollTop;
      if (seen.has(el)) scrollKick(top - seen.get(el));
      seen.set(el, top);
    },
    true,
  );
} catch {
  /* cross-origin host: no scroll input, the loop still runs */
}

// -------------------------------------------------------------------- load
new GLTFLoader().load("./baseball.glb", (gltf) => {
  const ball = gltf.scene;

  // normalise so the ball has radius 1 and sits on the origin
  const sphere = new THREE.Box3()
    .setFromObject(ball)
    .getBoundingSphere(new THREE.Sphere());
  ball.scale.setScalar(1 / sphere.radius);
  ball.position.sub(sphere.center.clone().multiplyScalar(1 / sphere.radius));

  ball.traverse((o) => {
    if (!o.isMesh) return;
    o.geometry.computeVertexNormals();
    o.material.metalness = 0; // a baseball is dielectric
    if (o.material.map) {
      o.material.map.anisotropy = renderer.capabilities.getMaxAnisotropy();
    }
  });

  spinner.add(ball);
});

// -------------------------------------------------------------------- loop
let t = 0;
let last = performance.now();
function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.1); // a backgrounded tab can
  last = now; // hand back a huge delta
  t += dt;
  integrateFling(dt);
  poseAt(t);
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ------------------------------------------------------------------ resize
function resize() {
  const w = document.body.clientWidth || 1;
  const h = document.body.clientHeight || 1;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(document.body);
resize();

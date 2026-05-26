// MediaPipe Tasks-Vision Web Worker (MODULE worker)
//
// Uses tasks-vision 0.10.35 (latest), which supports module workers
// without falling back to importScripts() internally.

import {
  FilesetResolver,
  HandLandmarker,
  FaceLandmarker,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/vision_bundle.mjs';

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm';
const HAND_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const FACE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

let vision = null;
let handLandmarker = null;
let faceLandmarker = null;
let lastHandTs = 0;
let lastFaceTs = 0;

async function ensureVision() {
  if (!vision) vision = await FilesetResolver.forVisionTasks(WASM_BASE);
  return vision;
}

async function initHands(opts = {}) {
  const v = await ensureVision();
  if (handLandmarker) { handLandmarker.close(); handLandmarker = null; }
  handLandmarker = await HandLandmarker.createFromOptions(v, {
    baseOptions: { modelAssetPath: HAND_MODEL, delegate: opts.delegate || 'GPU' },
    runningMode: 'VIDEO',
    numHands: opts.maxNumHands ?? 2,
    minHandDetectionConfidence: opts.minDetectionConfidence ?? 0.7,
    minHandPresenceConfidence: opts.minPresenceConfidence ?? 0.5,
    minTrackingConfidence: opts.minTrackingConfidence ?? 0.5,
  });
}

async function initFace(opts = {}) {
  const v = await ensureVision();
  if (faceLandmarker) { faceLandmarker.close(); faceLandmarker = null; }
  faceLandmarker = await FaceLandmarker.createFromOptions(v, {
    baseOptions: { modelAssetPath: FACE_MODEL, delegate: opts.delegate || 'GPU' },
    runningMode: 'VIDEO',
    numFaces: opts.maxNumFaces ?? 1,
    minFaceDetectionConfidence: opts.minDetectionConfidence ?? 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: opts.minTrackingConfidence ?? 0.5,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
}

function toLegacyHands(result) {
  const handedness = (result.handedness || []).map((h) => {
    const c = h[0] || {};
    return { label: c.categoryName, score: c.score, index: c.index };
  });
  return {
    multiHandLandmarks: result.landmarks || [],
    multiHandWorldLandmarks: result.worldLandmarks || [],
    multiHandedness: handedness,
  };
}

function toLegacyFace(result) {
  return { multiFaceLandmarks: result.faceLandmarks || [] };
}

self.onmessage = async (e) => {
  const { type, payload, reqId } = e.data || {};
  try {
    if (type === 'init-hands') {
      await initHands(payload);
      self.postMessage({ type: 'ready', target: 'hands', reqId });
    } else if (type === 'init-face') {
      await initFace(payload);
      self.postMessage({ type: 'ready', target: 'face', reqId });
    } else if (type === 'stop-hands') {
      if (handLandmarker) { handLandmarker.close(); handLandmarker = null; }
    } else if (type === 'stop-face') {
      if (faceLandmarker) { faceLandmarker.close(); faceLandmarker = null; }
    } else if (type === 'frame') {
      const { bitmap, ts, runHands, runFace } = payload;
      if (runHands && handLandmarker) {
        const safeTs = ts > lastHandTs ? ts : lastHandTs + 1;
        lastHandTs = safeTs;
        const r = handLandmarker.detectForVideo(bitmap, safeTs);
        self.postMessage({ type: 'hands-result', payload: toLegacyHands(r) });
      }
      if (runFace && faceLandmarker) {
        const safeTs = ts > lastFaceTs ? ts : lastFaceTs + 1;
        lastFaceTs = safeTs;
        const r = faceLandmarker.detectForVideo(bitmap, safeTs);
        self.postMessage({ type: 'face-result', payload: toLegacyFace(r) });
      }
      if (bitmap && typeof bitmap.close === 'function') bitmap.close();
    }
  } catch (err) {
    self.postMessage({
      type: 'error',
      payload: { message: String(err && err.message || err), stack: err && err.stack },
    });
  }
};

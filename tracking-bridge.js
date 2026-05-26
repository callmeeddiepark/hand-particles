// Tracking Bridge — abstraction over MediaPipe tracking backends.
//
// Backends (chosen via URL flags):
//   default        — Tasks-Vision on main thread (GPU + lightweight float16 models)
//   ?worker=1      — Tasks-Vision in a module Web Worker (currently broken upstream:
//                    tasks-vision calls importScripts() inside a module worker)
//   ?legacy=1      — Legacy @mediapipe/hands + @mediapipe/face_mesh (rollback)
//   ?cpu           — Force CPU delegate (debug)
//
// Public API (window.Tracking):
//   init()                          → Promise<void>
//   attachHands(opts, onResults)    → Promise<void>
//   detachHands()
//   attachFace(opts, onResults)     → Promise<void>
//   detachFace()
//   bindVideo(videoEl)              starts the internal send loop
//   unbindVideo()                   stops the send loop
//   isLegacy, isWorker, isMain      flags
//
// Result callbacks receive the LEGACY result shape:
//   hands: { multiHandLandmarks, multiHandedness, multiHandWorldLandmarks }
//   face:  { multiFaceLandmarks }
// so existing onResults code in index.html does not need to change.

(function () {
  const params = new URLSearchParams(location.search);
  const FORCE_LEGACY = params.has('legacy');
  // Default: Tasks-Vision on main thread (GPU + lightweight models).
  // Worker mode is opt-in via ?worker=1 — currently broken upstream due to
  // tasks-vision calling importScripts() inside a module worker.
  const USE_WORKER = params.has('worker') && !FORCE_LEGACY && typeof Worker !== 'undefined';
  const NO_WORKER = !FORCE_LEGACY && !USE_WORKER;
  const FORCE_CPU = params.has('cpu');

  const TASKS_VISION_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/vision_bundle.mjs';
  const TASKS_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm';
  const HAND_MODEL =
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
  const FACE_MODEL =
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

  let worker = null;
  let initPromise = null;

  // Main-thread Tasks-Vision instances (NO_WORKER path)
  let visionMain = null;
  let mainHand = null;
  let mainFace = null;
  let mainLastHandTs = 0;
  let mainLastFaceTs = 0;

  // Legacy path instances (FORCE_LEGACY path)
  let legacyHands = null;
  let legacyFace = null;

  // Shared state
  let handsCb = null;
  let faceCb = null;
  let handsActive = false;
  let faceActive = false;
  let videoEl = null;
  let loopId = null;
  let pendingFrame = false;
  let frameSerial = 0;

  // ---------- WORKER PATH ----------

  function ensureWorker() {
    if (worker) return worker;
    worker = new Worker('./tracking-worker.js?v=3', { type: 'module' });
    worker.onmessage = (e) => {
      const { type, payload } = e.data || {};
      if (type === 'hands-result') {
        if (handsCb && handsActive) handsCb(payload);
      } else if (type === 'face-result') {
        if (faceCb && faceActive) faceCb(payload);
      } else if (type === 'error') {
        const msg = (payload && payload.message) || String(payload);
        const stack = (payload && payload.stack) || '';
        console.error('[tracking-worker] ERROR:', msg);
        if (stack) console.error('[tracking-worker] STACK:', stack);
      }
    };
    worker.onerror = (e) => {
      console.error('[tracking-worker] fatal:', e.message || e.filename + ':' + e.lineno || e);
    };
    return worker;
  }

  function workerSend(msg, transfer) {
    ensureWorker().postMessage(msg, transfer || []);
  }

  // ---------- MAIN-THREAD TASKS-VISION PATH ----------

  async function ensureVisionMain() {
    if (visionMain) return visionMain;
    const mod = await import(TASKS_VISION_URL);
    const fileset = await mod.FilesetResolver.forVisionTasks(TASKS_WASM);
    visionMain = { mod, fileset };
    return visionMain;
  }

  async function initHandsMain(opts) {
    const { mod, fileset } = await ensureVisionMain();
    if (mainHand) { mainHand.close(); mainHand = null; }
    mainHand = await mod.HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: FORCE_CPU ? 'CPU' : 'GPU' },
      runningMode: 'VIDEO',
      numHands: opts.maxNumHands ?? 2,
      minHandDetectionConfidence: opts.minDetectionConfidence ?? 0.7,
      minHandPresenceConfidence: opts.minPresenceConfidence ?? 0.5,
      minTrackingConfidence: opts.minTrackingConfidence ?? 0.5,
    });
  }

  async function initFaceMain(opts) {
    const { mod, fileset } = await ensureVisionMain();
    if (mainFace) { mainFace.close(); mainFace = null; }
    mainFace = await mod.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: FORCE_CPU ? 'CPU' : 'GPU' },
      runningMode: 'VIDEO',
      numFaces: opts.maxNumFaces ?? 1,
      minFaceDetectionConfidence: opts.minDetectionConfidence ?? 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: opts.minTrackingConfidence ?? 0.5,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });
  }

  function toLegacyHands(r) {
    const handedness = (r.handedness || []).map((h) => {
      const c = h[0] || {};
      return { label: c.categoryName, score: c.score, index: c.index };
    });
    return {
      multiHandLandmarks: r.landmarks || [],
      multiHandWorldLandmarks: r.worldLandmarks || [],
      multiHandedness: handedness,
    };
  }
  function toLegacyFace(r) {
    return { multiFaceLandmarks: r.faceLandmarks || [] };
  }

  // ---------- LEGACY PATH ----------

  function ensureLegacyHands(opts, cb) {
    if (legacyHands) return legacyHands;
    if (typeof window.Hands === 'undefined') {
      throw new Error('Legacy @mediapipe/hands not loaded');
    }
    legacyHands = new window.Hands({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${f}`,
    });
    legacyHands.setOptions({
      maxNumHands: opts.maxNumHands ?? 2,
      modelComplexity: opts.modelComplexity ?? 1,
      minDetectionConfidence: opts.minDetectionConfidence ?? 0.7,
      minTrackingConfidence: opts.minTrackingConfidence ?? 0.5,
    });
    legacyHands.onResults((r) => { if (cb && handsActive) cb(r); });
    return legacyHands;
  }

  function ensureLegacyFace(opts, cb) {
    if (legacyFace) return legacyFace;
    if (typeof window.FaceMesh === 'undefined') {
      throw new Error('Legacy @mediapipe/face_mesh not loaded');
    }
    legacyFace = new window.FaceMesh({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${f}`,
    });
    legacyFace.setOptions({
      maxNumFaces: opts.maxNumFaces ?? 1,
      refineLandmarks: !!opts.refineLandmarks,
      minDetectionConfidence: opts.minDetectionConfidence ?? 0.5,
      minTrackingConfidence: opts.minTrackingConfidence ?? 0.5,
    });
    legacyFace.onResults((r) => { if (cb && faceActive) cb(r); });
    return legacyFace;
  }

  // ---------- FRAME LOOP ----------

  function stopLoop() {
    if (loopId) { cancelAnimationFrame(loopId); loopId = null; }
    pendingFrame = false;
  }

  function startLoop() {
    if (loopId) return;
    const tick = async () => {
      if (!videoEl || (!handsActive && !faceActive)) { loopId = null; return; }
      if (videoEl.readyState >= 2 && !pendingFrame) {
        pendingFrame = true;
        const ts = performance.now();
        try {
          if (USE_WORKER) {
            const bitmap = await createImageBitmap(videoEl);
            workerSend(
              { type: 'frame', payload: { bitmap, ts, runHands: handsActive, runFace: faceActive } },
              [bitmap],
            );
          } else if (NO_WORKER) {
            if (handsActive && mainHand) {
              const t = ts > mainLastHandTs ? ts : mainLastHandTs + 1;
              mainLastHandTs = t;
              const r = mainHand.detectForVideo(videoEl, t);
              if (handsCb) handsCb(toLegacyHands(r));
            }
            if (faceActive && mainFace) {
              const t = ts > mainLastFaceTs ? ts : mainLastFaceTs + 1;
              mainLastFaceTs = t;
              const r = mainFace.detectForVideo(videoEl, t);
              if (faceCb) faceCb(toLegacyFace(r));
            }
          } else if (FORCE_LEGACY) {
            const tasks = [];
            if (handsActive && legacyHands) tasks.push(legacyHands.send({ image: videoEl }));
            if (faceActive && legacyFace) tasks.push(legacyFace.send({ image: videoEl }));
            if (tasks.length) await Promise.all(tasks);
          }
        } catch (e) {
          // swallow per-frame errors; tracking continues next frame
        }
        pendingFrame = false;
      }
      loopId = requestAnimationFrame(tick);
    };
    loopId = requestAnimationFrame(tick);
  }

  // ---------- PUBLIC API ----------

  const Tracking = {
    isLegacy: FORCE_LEGACY,
    isWorker: USE_WORKER,
    isMain: NO_WORKER,

    async init() {
      if (initPromise) return initPromise;
      initPromise = (async () => {
        if (USE_WORKER) ensureWorker();
      })();
      return initPromise;
    },

    async attachHands(opts, onResults) {
      await this.init();
      handsCb = onResults;
      if (FORCE_LEGACY) {
        ensureLegacyHands(opts || {}, onResults);
      } else if (NO_WORKER) {
        await initHandsMain(opts || {});
      } else {
        const payload = { ...(opts || {}) };
        if (FORCE_CPU) payload.delegate = 'CPU';
        workerSend({ type: 'init-hands', payload });
      }
      handsActive = true;
      if (videoEl) startLoop();
    },

    detachHands() {
      handsActive = false;
      handsCb = null;
      if (FORCE_LEGACY && legacyHands) {
        try { legacyHands.close && legacyHands.close(); } catch (_) {}
        legacyHands = null;
      } else if (NO_WORKER && mainHand) {
        mainHand.close(); mainHand = null;
      } else if (USE_WORKER) {
        workerSend({ type: 'stop-hands' });
      }
      if (!faceActive) stopLoop();
    },

    async attachFace(opts, onResults) {
      await this.init();
      faceCb = onResults;
      if (FORCE_LEGACY) {
        ensureLegacyFace(opts || {}, onResults);
      } else if (NO_WORKER) {
        await initFaceMain(opts || {});
      } else {
        const payload = { ...(opts || {}) };
        if (FORCE_CPU) payload.delegate = 'CPU';
        workerSend({ type: 'init-face', payload });
      }
      faceActive = true;
      if (videoEl) startLoop();
    },

    detachFace() {
      faceActive = false;
      faceCb = null;
      if (FORCE_LEGACY && legacyFace) {
        try { legacyFace.close && legacyFace.close(); } catch (_) {}
        legacyFace = null;
      } else if (NO_WORKER && mainFace) {
        mainFace.close(); mainFace = null;
      } else if (USE_WORKER) {
        workerSend({ type: 'stop-face' });
      }
      if (!handsActive) stopLoop();
    },

    bindVideo(v) {
      videoEl = v;
      if (handsActive || faceActive) startLoop();
    },

    unbindVideo() {
      videoEl = null;
      stopLoop();
    },
  };

  window.Tracking = Tracking;
})();

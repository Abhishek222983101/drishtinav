// Run the on-device JS engine on a recorded drive and compare against the Python reference.
//   python tests/js/export_drive.py > /tmp/drive.json && node tests/js/parity.mjs /tmp/drive.json
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const E = require('../../web/mobile/engine.js');
const drive = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const speednet = JSON.parse(readFileSync(new URL('../../models/speednet.json', import.meta.url)));
const roads = drive.roads ? JSON.parse(readFileSync(new URL(`../../data/osm/${drive.roads}`, import.meta.url))) : null;
const frame = new E.LocalFrame(drive.truth[0][0], drive.truth[0][1]);
const eng = new E.NavEngine({ speednet, roads, frame });
let errs = [], tunnelErr = null, lastDenied = -1, ms = 0;
for (let i = 0; i < drive.t.length; i++) {
  const g = drive.gnss[i];
  const fix = g ? { lat: g[0], lon: g[1], speed: g[2], course: g[3], accuracy: g[4] } : null;
  const st = eng.step(drive.t[i], drive.accel[i], drive.gyro[i], fix);
  ms += st.latencyMs;
  const [te, tn] = frame.toEN(drive.truth[i][0], drive.truth[i][1]);
  const e = Math.hypot(st.e - te, st.n - tn);
  errs.push(e);
  if (drive.denied[i]) lastDenied = i;
}
const sorted = [...errs].sort((a, b) => a - b);
const out = { samples: errs.length, median_err_m: sorted[Math.floor(sorted.length / 2)],
  denied_exit_err_m: lastDenied >= 0 ? errs[lastDenied] : null, max_err_m: sorted[sorted.length - 1],
  mean_step_ms: ms / errs.length, transitions: eng.handler.transitions.length, alignment: eng.align.state };
console.log(JSON.stringify(out, null, 1));

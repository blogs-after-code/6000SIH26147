// Vercel build step: writes js/config.js so the static front end knows where the Render backend is.
// Set API_BASE_URL in Vercel > Project > Settings > Environment Variables, e.g. https://sanketsetu-api.onrender.com
import { writeFileSync } from "node:fs";

const base = (process.env.API_BASE_URL || "").trim().replace(/\/+$/, "");
if (!/^https?:\/\//.test(base)) {
  console.error("API_BASE_URL is missing or not a full URL (needs https://...). Set it in Vercel > Settings > Environment Variables, then redeploy.");
  process.exit(1);
}
writeFileSync("js/config.js", `// Generated at build time by build-config.mjs. Do not edit.\nwindow.SANKETSETU_API_BASE = ${JSON.stringify(base)};\n`);
console.log("config.js written, backend =", base);

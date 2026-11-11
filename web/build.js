import fs from 'node:fs';
import path from 'node:path';

const configuredApiBase = process.env.VITE_API_URL || 'https://api.actualplate.tech';
let parsedApiBase;

try {
  parsedApiBase = new URL(configuredApiBase);
} catch {
  throw new Error('VITE_API_URL must be an absolute HTTP(S) URL.');
}

if (!['http:', 'https:'].includes(parsedApiBase.protocol)) {
  throw new Error('VITE_API_URL must use HTTP or HTTPS.');
}
if (parsedApiBase.username || parsedApiBase.password || parsedApiBase.search || parsedApiBase.hash) {
  throw new Error('VITE_API_URL must not contain credentials, query parameters, or a fragment.');
}

const apiBase = parsedApiBase.toString().replace(/\\/$/, '');
const templatePath = path.join(process.cwd(), 'index.html');
const outputDir = path.join(process.cwd(), 'dist');
const outputPath = path.join(outputDir, 'index.html');

const template = fs.readFileSync(templatePath, 'utf8');
const placeholder = '{{API_BASE_URL}}';
const placeholderCount = template.split(placeholder).length - 1;

if (placeholderCount !== 1) {
  throw new Error(`Expected exactly one ${placeholder} placeholder; found ${placeholderCount}.`);
}

const html = template.replace(placeholder, apiBase);
if (html.includes('{{API_') || html.includes('undefined/upload-image-stream')) {
  throw new Error('Build output contains an unresolved API configuration placeholder.');
}

fs.mkdirSync(outputDir, { recursive: true });
fs.writeFileSync(outputPath, html, 'utf8');
console.log(`Built web/dist/index.html with API base ${apiBase}`);

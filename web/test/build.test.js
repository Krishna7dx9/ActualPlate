import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const outputPath = path.join(process.cwd(), 'dist', 'index.html');
const html = fs.readFileSync(outputPath, 'utf8');

test('build injects a configured API base URL and removes template placeholders', () => {
  assert.match(html, /const API_BASE_URL = "https:\/\/api\\.example\\.test";/);
  assert.doesNotMatch(html, /\{\{API_[A-Z_]+\}\}/);
  assert.match(html, /\/upload-image-stream/);
});

test('landing page explains B2B platform status without promising unshipped features', () => {
  assert.match(html, /B2B readiness in progress/);
  assert.match(html, /not yet a finished SaaS/i);
  assert.match(html, /does not currently claim a validated ≤5%/);
  assert.match(html, /Organization-scoped API keys/);
});

test('FatSecret attribution and terms links are present', () => {
  assert.match(html, /href="https:\/\/platform\.fatsecret\.com">Powered by fatsecret Platform API<\/a>/);
  assert.match(html, /https:\/\/platform\.fatsecret\.com\/terms/);
  assert.match(html, /https:\/\/platform\.fatsecret\.com\/attribution/);
});

test('demo handles missing numerical values without converting them to zero', () => {
  assert.match(html, /Not available/);
  assert.match(html, /Number\.isFinite\(value\)/);
});

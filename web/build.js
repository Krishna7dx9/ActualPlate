import fs from 'fs';
import path from 'path';

// Fallback to production API URL if not set in Netlify env
const apiUrl = process.env.VITE_API_URL || "https://api.actualplate.tech";

// Since netlify.toml has base="web", cwd is the web folder
const templatePath = path.join(process.cwd(), 'index.html');
const outputPath = path.join(process.cwd(), 'dist', 'index.html');

console.log(`Injecting API_URL: ${apiUrl}`);
const html = fs.readFileSync(templatePath, 'utf8');
const newHtml = html.replace('{{API_URL}}', apiUrl);

if (!fs.existsSync('dist')) fs.mkdirSync('dist');
fs.writeFileSync(outputPath, newHtml);
console.log('✅ Built index.html successfully');
import fs from 'fs';
import path from 'path';

// Reads from Netlify Env Vars OR local .env
const apiUrl = process.env.VITE_API_URL || process.env.API_URL; 

if (!apiUrl) {
    console.error("ERROR: VITE_API_URL environment variable is missing.");
    process.exit(1);
}

const templatePath = path.join(process.cwd(), 'web', 'index.html');
const outputPath = path.join(process.cwd(), 'dist', 'index.html');

console.log(`Injecting API_URL: ${apiUrl}`);
const html = fs.readFileSync(templatePath, 'utf8');
const newHtml = html.replace('{{API_URL}}', apiUrl);

if (!fs.existsSync('dist')) fs.mkdirSync('dist');
fs.writeFileSync(outputPath, newHtml);
console.log('✅ Built index.html');

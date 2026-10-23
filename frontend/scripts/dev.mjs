import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import net from 'node:net';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const frontendRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const projectRoot = resolve(frontendRoot, '..');
const pythonPath = resolve(
  projectRoot,
  '.venv',
  process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python',
);
const apiHost = '127.0.0.1';
const apiPort = 8000;
const viteCli = resolve(frontendRoot, 'node_modules/vite/bin/vite.js');

let backendProcess;
let frontendProcess;
let ownsBackend = false;
let shuttingDown = false;
let backendStartError;

function sleep(milliseconds) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds));
}

async function apiIsRunning() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 1000);
  try {
    const response = await fetch(`http://${apiHost}:${apiPort}/health`, {
      signal: controller.signal,
    });
    if (!response.ok) return false;
    const body = await response.json();
    return body.service === 'Plant Guard AI API';
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

function portIsOpen() {
  return new Promise((resolvePromise) => {
    const socket = net.createConnection({ host: apiHost, port: apiPort });
    const finish = (isOpen) => {
      socket.destroy();
      resolvePromise(isOpen);
    };
    socket.setTimeout(500);
    socket.once('connect', () => finish(true));
    socket.once('error', () => finish(false));
    socket.once('timeout', () => finish(false));
  });
}

function stopProcesses() {
  if (shuttingDown) return;
  shuttingDown = true;
  if (frontendProcess && frontendProcess.exitCode === null) {
    frontendProcess.kill();
  }
  if (ownsBackend && backendProcess && backendProcess.exitCode === null) {
    backendProcess.kill();
  }
}

process.once('SIGINT', () => {
  stopProcesses();
  process.exitCode = 130;
});
process.once('SIGTERM', () => {
  stopProcesses();
  process.exitCode = 143;
});

async function waitForApi() {
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    if (await apiIsRunning()) return;
    if (backendStartError) {
      throw new Error(`Could not start backend: ${backendStartError.message}`);
    }
    if (backendProcess?.exitCode !== null && backendProcess?.exitCode !== undefined) {
      throw new Error(
        `Backend exited with code ${backendProcess.exitCode}. Check its startup output above.`,
      );
    }
    await sleep(500);
  }
  throw new Error('Backend did not become available within 120 seconds.');
}

async function main() {
  if (!existsSync(viteCli)) {
    throw new Error('Frontend packages are missing. Run `npm ci` in the frontend folder.');
  }

  if (await apiIsRunning()) {
    console.log(`Using existing Plant Guard API at http://${apiHost}:${apiPort}`);
  } else {
    if (await portIsOpen()) {
      throw new Error(
        `Port ${apiPort} is already in use by another service. Stop it before starting Plant Guard.`,
      );
    }
    if (!existsSync(pythonPath)) {
      throw new Error(
        'Project Python environment not found. Follow the Windows setup in the repository README first.',
      );
    }

    console.log('Starting the Plant Guard backend (model startup can take a few seconds)…');
    ownsBackend = true;
    backendProcess = spawn(
      pythonPath,
      [
        '-m',
        'uvicorn',
        'backend.app.main:app',
        '--host',
        apiHost,
        '--port',
        String(apiPort),
      ],
      { cwd: projectRoot, stdio: 'inherit', env: process.env },
    );
    backendProcess.once('error', (error) => {
      backendStartError = error;
      stopProcesses();
      process.exitCode = 1;
    });

    await waitForApi();
    console.log(`Backend ready at http://${apiHost}:${apiPort}`);
  }

  console.log('Starting the Plant Guard frontend…');
  frontendProcess = spawn(process.execPath, [viteCli, '--host', '0.0.0.0'], {
    cwd: frontendRoot,
    stdio: 'inherit',
    env: process.env,
  });
  frontendProcess.once('error', (error) => {
    console.error(`Could not start frontend: ${error.message}`);
    stopProcesses();
    process.exitCode = 1;
  });
  frontendProcess.once('exit', (code) => {
    if (!shuttingDown) {
      console.error(`Frontend exited with code ${code ?? 'unknown'}.`);
      stopProcesses();
      process.exitCode = code ?? 1;
    }
  });
  backendProcess?.once('exit', (code) => {
    if (!shuttingDown && ownsBackend) {
      console.error(`Backend exited with code ${code ?? 'unknown'}.`);
      stopProcesses();
      process.exitCode = code ?? 1;
    }
  });
}

main().catch((error) => {
  console.error(`\nPlant Guard could not start: ${error.message}`);
  stopProcesses();
  process.exitCode = 1;
});

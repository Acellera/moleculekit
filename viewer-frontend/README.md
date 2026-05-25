# moleculekit viewer frontend

Source for the mol*-based browser viewer driven by `Molecule.view(viewer="molstar")`. Built output lands in `../moleculekit/viewer/molstar/static/` and is committed alongside this source — installing moleculekit does **not** run `npm install`.

## Build

```
cd viewer-frontend
npm install
npm run build
```

Output goes to `../moleculekit/viewer/molstar/static/`. Commit both this folder and the static output when you change the frontend.

## Dev server

`npm run dev` runs Vite. To exercise the Python server alongside Vite, point the dev server at the Python backend with a proxy (see `vite.config.ts`).

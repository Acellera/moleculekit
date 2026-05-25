import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import license from "rollup-plugin-license";
import path from "node:path";

// We bundle mol*, React, etc. locally so the viewer works offline. Their
// MIT/Apache/BSD licenses require we include the copyright + license text
// alongside the shipped artifact: rollup-plugin-license walks the bundled
// dependency tree and emits THIRD_PARTY_LICENSES.txt next to the JS.
export default defineConfig({
  plugins: [
    react(),
    license({
      thirdParty: {
        output: {
          file: path.resolve(
            __dirname,
            "../moleculekit/viewer/molstar/static/THIRD_PARTY_LICENSES.txt"
          ),
        },
      },
    }),
  ],
  base: "./",
  build: {
    outDir: path.resolve(__dirname, "../moleculekit/viewer/molstar/static"),
    emptyOutDir: true,
    sourcemap: false,
  },
});

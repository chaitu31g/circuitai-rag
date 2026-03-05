import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "0.0.0.0",        // allow external access (ngrok, Colab port-forwarding)
    port: 5173,
    allowedHosts: "all",    // accept requests from any host (ngrok tunnels, etc.)
    cors: true,
  },
});
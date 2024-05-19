/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */
await import("./src/env.js");

/** @type {import("next").NextConfig} */
const config = {
   reactStrictMode: true,
   images: {
     domains: ['lh3.googleusercontent.com'],
     remotePatterns: [
      {
        protocol: "https",
        hostname: "**",
      },
    ],
     
   },

};

export default config;


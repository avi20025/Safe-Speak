// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack(config) {
    // Exclude SVGs from default asset loader
    config.module.rules = config.module.rules.map((rule) => {
      if (rule.test && rule.test.toString().includes('svg')) {
        return { ...rule, exclude: /\.svg$/i };
      }
      return rule;
    });

    // Add SVGR for importing SVGs as components
    config.module.rules.push({
      test: /\.svg$/,
      issuer: /\.[jt]sx?$/,
      use: ['@svgr/webpack'],
    });

    return config;
  },
};

module.exports = nextConfig;

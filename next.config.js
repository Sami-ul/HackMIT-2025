/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config) => {
    // Handle deck.gl peer dependencies
    config.resolve.alias = {
      ...config.resolve.alias,
      '@deck.gl/core': require.resolve('@deck.gl/core'),
      '@deck.gl/layers': require.resolve('@deck.gl/layers'),
      '@deck.gl/react': require.resolve('@deck.gl/react'),
    };
    return config;
  },
}

module.exports = nextConfig
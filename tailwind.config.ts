import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0b0c0d",
          900: "#101112",
          850: "#151617",
        },
        lumenci: "#f47742",
      },
    },
  },
  plugins: [],
};

export default config;

import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      colors: {
        ink: {
          DEFAULT: "#05070A",
          card: "#0D1117",
          muted: "#161b22",
        },
        accent: {
          purple: "#6D5EF7",
          blue: "#4F8CFF",
        },
        stone: {
          cream: "#fafafa",
          muted: "#a1a1aa",
        },
      },
      boxShadow: {
        card: "0 1px 0 rgba(255,255,255,0.03) inset, 0 4px 24px rgba(0,0,0,0.35)",
        lift: "0 8px 32px rgba(0,0,0,0.4)",
      },
    },
  },
  plugins: [],
};

export default config;

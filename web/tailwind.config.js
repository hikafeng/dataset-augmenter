// filepath: \home\hika\work\Dataset-Augmenter\web\tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./styles/**/*.{css}"
  ],
  theme: {
    extend: {
      // 根据设计需求添加自定义颜色、间距等
    },
  },
  plugins: [
    require('@tailwindcss/typography')
  ],
}
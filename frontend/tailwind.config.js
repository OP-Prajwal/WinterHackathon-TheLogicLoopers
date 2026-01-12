/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                dark: {
                    900: '#0a0a0a',
                    800: '#121212',
                    700: '#1a1a1a',
                },
                cyan: {
                    glow: '#00f0ff',
                },
                violet: {
                    glow: '#7000ff',
                },
            },
            fontFamily: {
                mono: ['"JetBrains Mono"', 'monospace'],
                sans: ['"Inter"', 'sans-serif'],
            },
            backgroundImage: {
                'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
                'cyber-grid': "url(\"data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0h40v40H0V0zm1 1h38v38H1V1z' fill='%23222' fill-opacity='0.1' fill-rule='evenodd'/%3E%3C/svg%3E\")",
            },
            boxShadow: {
                'neon-cyan': '0 0 5px theme("colors.cyan.400"), 0 0 20px theme("colors.cyan.700")',
                'neon-violet': '0 0 5px theme("colors.violet.400"), 0 0 20px theme("colors.violet.700")',
            },
        },
    },
    plugins: [],
}

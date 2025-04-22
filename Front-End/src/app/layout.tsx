import type { Metadata } from "next";

import "./globals.css";

// app/layout.tsx
import { Poppins } from 'next/font/google'

const poppins = Poppins({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-poppins',
})

export const metadata = {
  title: 'SafeSpeak',
  description: 'Voice verification you can trust',
}
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={poppins.variable}>
      <body className="font-poppins bg-gradient-to-b from-[#004AAC] to-[#0096DF]  ">
        {children}
      </body>
    </html>
  )
}

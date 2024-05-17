'use client';
import "~/styles/globals.css";

import { SessionProvider } from 'next-auth/react';

import { GeistSans } from "geist/font/sans";



export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <SessionProvider>
      <body className={GeistSans.className}>{children}</body>
      </SessionProvider>
    </html>
  );
}

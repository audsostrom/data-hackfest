'use client';
import "~/styles/globals.css";

import { SessionProvider } from 'next-auth/react';

import { GeistSans } from "geist/font/sans";
import {AppRouterCacheProvider} from "@mui/material-nextjs/v13-appRouter";
import {ThemeProvider} from "@mui/material";
import theme from "~/app/theme";



export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
    <AppRouterCacheProvider>
      <SessionProvider>
        <ThemeProvider theme={theme}>
          <body className={GeistSans.className}>{children}</body>
        </ThemeProvider>
      </SessionProvider>
    </AppRouterCacheProvider>
    </html>
  );
}

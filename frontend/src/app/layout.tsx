'use client';
import "~/styles/globals.css";

import { SessionProvider } from 'next-auth/react';

import { GeistSans } from "geist/font/sans";
import {AppRouterCacheProvider} from "@mui/material-nextjs/v13-appRouter";
import {ThemeProvider} from "@mui/material";
import theme from "~/app/theme";
import Navbar from "./components/navbar/navbar";



export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
    <AppRouterCacheProvider>
      <body>
      <SessionProvider>
        <ThemeProvider theme={theme}>
        <Navbar></Navbar>
        {children}
        </ThemeProvider>
      </SessionProvider>

      </body>
    </AppRouterCacheProvider>
    </html>
  );
}

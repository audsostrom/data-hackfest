'use client';
import "~/styles/globals.css";


import { GeistSans } from "geist/font/sans";
import {AppRouterCacheProvider} from "@mui/material-nextjs/v13-appRouter";
import {ThemeProvider} from "@mui/material";
import theme from "~/app/theme";
import Navbar from "./_components/navbar/navbar";

import { TRPCReactProvider } from "~/trpc/react";
import { SessionProvider } from "next-auth/react";
import CssBaseline from "@mui/material/CssBaseline";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
    <AppRouterCacheProvider>
        <TRPCReactProvider>
          <SessionProvider>
            <ThemeProvider theme={theme}>
              <body className={GeistSans.className}>
              <CssBaseline />
              {children}
              </body>
            </ThemeProvider>
          </SessionProvider>
        </TRPCReactProvider>
    </AppRouterCacheProvider>
    </html>
  );
}







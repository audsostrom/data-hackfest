'use client';
import "~/styles/globals.css";


import { GeistSans } from "geist/font/sans";
import {AppRouterCacheProvider} from "@mui/material-nextjs/v13-appRouter";
import {ThemeProvider} from "@mui/material";
import theme from "~/app/theme";

import { TRPCReactProvider } from "~/trpc/react";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
    <AppRouterCacheProvider>
        <TRPCReactProvider>
          <ThemeProvider theme={theme}>
            <body className={GeistSans.className}>{children}</body>
          </ThemeProvider>
        </TRPCReactProvider>
    </AppRouterCacheProvider>
    </html>
  );
}







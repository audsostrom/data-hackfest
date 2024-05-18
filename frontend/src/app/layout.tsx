import "~/styles/globals.css";


import { GeistSans } from "geist/font/sans";
import {AppRouterCacheProvider} from "@mui/material-nextjs/v13-appRouter";
import {ThemeProvider} from "@mui/material";
import theme from "~/app/theme";
import Navbar from "./_components/navbar/navbar";

import { TRPCReactProvider } from "~/trpc/react";
import { SessionProvider } from "next-auth/react";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
    <AppRouterCacheProvider>
      <body>
      <TRPCReactProvider>
        <ThemeProvider theme={theme}>
        {children}
        </ThemeProvider>
        </TRPCReactProvider>

      </body>
    </AppRouterCacheProvider>
    </html>
  );
}







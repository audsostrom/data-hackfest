import Box from "@mui/material/Box";

export default function LoginLayout({ children }: { children: React.ReactNode }) {
  return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
        }}
      >
        { children }
      </Box>
  );
}